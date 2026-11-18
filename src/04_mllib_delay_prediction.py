from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import RegressionEvaluator, BinaryClassificationEvaluator

spark = (
    SparkSession.builder
    .appName("SupplyChainMLlibPrediction")
    .config("spark.driver.memory", "2g")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

# 1. Carregar Silver Shipments + Features Topológicas Gold (GraphFrames)
shipments = spark.read.parquet("data/processed/silver_shipments.parquet")
centrality = spark.read.parquet("data/processed/gold_node_centrality.parquet")

src_graph = centrality.select(
    F.col("id").alias("src"),
    F.col("pagerank_score").alias("src_pagerank"),
    F.col("out_degree").alias("src_out_degree")
)

dst_graph = centrality.select(
    F.col("id").alias("dst"),
    F.col("pagerank_score").alias("dst_pagerank"),
    F.col("in_degree").alias("dst_in_degree")
)

df = (
    shipments
    .join(src_graph, on="src", how="left")
    .join(dst_graph, on="dst", how="left")
    .withColumn("is_peak_season", F.when(F.col("month").isin([11, 12]), 1.0).otherwise(0.0))
    .fillna(0.0)
)

# 2. Engenharia de Features com Pipeline Spark MLlib
carrier_indexer = StringIndexer(inputCol="carrier", outputCol="carrier_idx", handleInvalid="keep")
carrier_encoder = OneHotEncoder(inputCols=["carrier_idx"], outputCols=["carrier_vec"])

feature_cols = [
    "distance_km",
    "cargo_tons",
    "weather_severity",
    "is_peak_season",
    "src_pagerank",
    "dst_pagerank",
    "src_out_degree",
    "dst_in_degree",
    "carrier_vec"
]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# 3. Modelo 1: Regressão de Horas de Atraso (GBTRegressor)
gbt_reg = GBTRegressor(featuresCol="features", labelCol="delay_hours", maxIter=40, maxDepth=5, seed=42)
reg_pipeline = Pipeline(stages=[carrier_indexer, carrier_encoder, assembler, gbt_reg])
reg_model = reg_pipeline.fit(train_df)
reg_preds = reg_model.transform(test_df).withColumnRenamed("prediction", "predicted_delay_h")

rmse = RegressionEvaluator(labelCol="delay_hours", predictionCol="predicted_delay_h", metricName="rmse").evaluate(reg_preds)
r2 = RegressionEvaluator(labelCol="delay_hours", predictionCol="predicted_delay_h", metricName="r2").evaluate(reg_preds)

print("\n🤖 AVALIAÇÃO DO MODELO DE REGRESSÃO DE ATRASO (GBTRegressor - MLlib):")
print(f"   -> RMSE (Erro Quadrático Médio): {rmse:.3f} horas")
print(f"   -> R² (Variância Explicada):     {r2:.4f}")

# 4. Modelo 2: Classificação de Risco de Quebra de SLA (RandomForestClassifier)
rf_clf = RandomForestClassifier(featuresCol="features", labelCol="sla_breached", numTrees=60, maxDepth=6, seed=42)
clf_pipeline = Pipeline(stages=[carrier_indexer, carrier_encoder, assembler, rf_clf])
clf_model = clf_pipeline.fit(train_df)
clf_preds = clf_model.transform(test_df)

auc_roc = BinaryClassificationEvaluator(labelCol="sla_breached", rawPredictionCol="rawPrediction", metricName="areaUnderROC").evaluate(clf_preds)

print("\n🎯 AVALIAÇÃO DO MODELO DE CLASSIFICAÇÃO DE RISCO SLA (RandomForest - MLlib):")
print(f"   -> AUC-ROC: {auc_roc:.4f}")

# Extrair Feature Importances do RandomForest
rf_stage = clf_model.stages[-1]
importances = rf_stage.featureImportances.toArray()
expanded_names = [
    "distance_km", "cargo_tons", "weather_severity", "is_peak_season",
    "src_pagerank (GraphFrames)", "dst_pagerank (GraphFrames)",
    "src_out_degree (GraphFrames)", "dst_in_degree (GraphFrames)",
    "carrier_1", "carrier_2", "carrier_3", "carrier_4"
]

print("\n📈 TOP FEATURES MAIS IMPORTANTES PARA PREVER QUEBRA DE SLA:")
feat_imp = sorted(zip(expanded_names, importances), key=lambda x: x[1], reverse=True)
for name, score in feat_imp[:8]:
    print(f"   - {name:<30}: {score:.4f}")

# 5. Consolidar Predições na Camada Gold
gold_predictions = (
    reg_preds
    .select(
        "shipment_id", "dispatch_date", "src", "dst", "carrier",
        "cargo_tons", "weather_severity", "delay_hours",
        F.round("predicted_delay_h", 2).alias("predicted_delay_h"),
        "sla_breached"
    )
)

gold_predictions.write.mode("overwrite").parquet("data/processed/gold_shipment_predictions.parquet")

print("\n🔮 AMOSTRA DE PREDIÇÕES NA CAMADA GOLD (REAL VS PREVISTO):")
gold_predictions.orderBy(F.desc("predicted_delay_h")).show(10, truncate=False)

spark.stop()
