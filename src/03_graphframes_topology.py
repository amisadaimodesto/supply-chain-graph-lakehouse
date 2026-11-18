import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from graphframes import GraphFrame

os.makedirs("/tmp/spark-checkpoints", exist_ok=True)

spark = (
    SparkSession.builder
    .appName("SupplyChainGraphFramesTopology")
    .config("spark.driver.memory", "2g")
    .config("spark.sql.adaptive.enabled", "false")
    .config("spark.jars.packages", "graphframes:graphframes:0.8.3-spark3.5-s_2.12")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")
spark.sparkContext.setCheckpointDir("/tmp/spark-checkpoints")

# 1. Carregar Vértices (Bronze) e Arestas Enriquecidas (Silver)
vertices = spark.read.parquet("data/raw/nodes.parquet")
routes = spark.read.parquet("data/raw/routes.parquet")
route_metrics = spark.read.parquet("data/processed/silver_route_metrics.parquet")

edges = (
    routes.join(route_metrics, on=["src", "dst"], how="left")
    .fillna(0)
)

g = GraphFrame(vertices, edges)

# 2. PageRank para identificar nós críticos (gargalos estruturais)
pr_results = g.pageRank(resetProbability=0.15, maxIter=15)
pagerank_df = pr_results.vertices.select(
    "id", "name", "node_type", "state", "capacity_tons",
    F.round("pagerank", 4).alias("pagerank_score")
)

# 3. Graus de Entrada (In-Degree) e Saída (Out-Degree)
in_deg = g.inDegrees.withColumnRenamed("inDegree", "in_degree")
out_deg = g.outDegrees.withColumnRenamed("outDegree", "out_degree")

# 4. Shortest Paths até hubs e lojas críticas
landmarks = ["DC_01", "DC_02", "STR_01", "STR_07"]
sp_results = g.shortestPaths(landmarks=landmarks)
sp_df = sp_results.select(
    "id",
    F.coalesce(F.col("distances")["DC_01"], F.lit(-1)).alias("hops_to_dc01"),
    F.coalesce(F.col("distances")["DC_02"], F.lit(-1)).alias("hops_to_dc02"),
    F.coalesce(F.col("distances")["STR_01"], F.lit(-1)).alias("hops_to_str01"),
)

# 5. Consolidar Tabela Gold de Centralidade Topológica
gold_node_centrality = (
    pagerank_df
    .join(in_deg, on="id", how="left")
    .join(out_deg, on="id", how="left")
    .join(sp_df, on="id", how="left")
    .fillna({"in_degree": 0, "out_degree": 0})
    .orderBy(F.desc("pagerank_score"))
)

gold_node_centrality.write.mode("overwrite").parquet("data/processed/gold_node_centrality.parquet")

print("\n🕸️ RANKING DE CRITICIDADE NA MALHA LOGÍSTICA (PAGERANK & GRAUS - GRAPHFRAMES):")
gold_node_centrality.show(12, truncate=False)

# 6. Simulação de Resiliência: O que acontece se DC_01 (Campinas) sair do ar?
print("⚠️ SIMULAÇÃO DE CONTINGÊNCIA (REMOÇÃO DO HUB CRÍTICO DC_01):")
g_contingencia = g.filterVertices("id != 'DC_01'")
cc = g_contingencia.connectedComponents(algorithm="graphx")

cc_summary = (
    cc.groupBy("component")
    .agg(
        F.count("*").alias("qtd_nos"),
        F.collect_list("id").alias("nos_no_componente")
    )
    .orderBy(F.desc("qtd_nos"))
)
cc_summary.show(truncate=False)

# 7. Motif Finding: Identificar rotas de abastecimento de 2 saltos (Fábrica -> CD -> Loja) com alto atraso
print("🔍 MOTIF FINDING (FÁBRICA -> CD -> LOJA COM ATRASO MÉDIO > 2.0h NO TRECHO FINAL):")
motifs = g.find("(fac)-[e1]->(dc); (dc)-[e2]->(store)")
critical_chains = (
    motifs
    .filter("fac.node_type = 'FACTORY' AND dc.node_type = 'DC' AND store.node_type = 'STORE'")
    .filter("e2.avg_delay_h > 2.0")
    .select(
        F.col("fac.id").alias("fabrica"),
        F.col("dc.id").alias("cd_intermediario"),
        F.col("store.id").alias("loja_destino"),
        F.round(F.col("e1.avg_actual_lead_time_h") + F.col("e2.avg_actual_lead_time_h"), 2).alias("lead_time_total_h"),
        F.col("e2.sla_breach_pct").alias("quebra_sla_ultimo_trecho_pct")
    )
    .orderBy(F.desc("quebra_sla_ultimo_trecho_pct"))
)
critical_chains.show(10, truncate=False)

spark.stop()
