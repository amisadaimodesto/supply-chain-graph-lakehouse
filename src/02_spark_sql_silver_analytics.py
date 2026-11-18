from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("SupplyChainSparkSQLSilver")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

# Leitura da Camada Bronze
nodes = spark.read.parquet("data/raw/nodes.parquet")
routes = spark.read.parquet("data/raw/routes.parquet")
shipments = spark.read.parquet("data/raw/shipments.parquet")

nodes.createOrReplaceTempView("nodes")
routes.createOrReplaceTempView("routes")
shipments.createOrReplaceTempView("shipments")

# 1. Enriquecimento Silver via Spark SQL (Joins + Window Functions)
silver_shipments = spark.sql("""
    SELECT
        s.shipment_id,
        TO_DATE(s.dispatch_date) AS dispatch_date,
        s.month,
        s.day_of_week,
        s.src,
        n_src.name AS src_name,
        n_src.node_type AS src_type,
        n_src.state AS src_state,
        s.dst,
        n_dst.name AS dst_name,
        n_dst.node_type AS dst_type,
        n_dst.state AS dst_state,
        s.carrier,
        s.cargo_tons,
        s.weather_severity,
        s.distance_km,
        s.std_lead_time_h,
        s.sla_hours,
        s.actual_lead_time_h,
        s.delay_hours,
        s.sla_breached,
        s.freight_cost_brl,
        ROUND(s.freight_cost_brl / s.cargo_tons, 2) AS cost_per_ton,
        ROUND(
            AVG(s.delay_hours) OVER (
                PARTITION BY s.src, s.dst
                ORDER BY TO_DATE(s.dispatch_date)
                ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING
            ), 2
        ) AS rolling_30_route_delay_h
    FROM shipments s
    JOIN nodes n_src ON s.src = n_src.id
    JOIN nodes n_dst ON s.dst = n_dst.id
""")

silver_shipments.write.mode("overwrite").parquet("data/processed/silver_shipments.parquet")
silver_shipments.createOrReplaceTempView("silver_shipments")

# 2. Métricas Consolidadas por Rota para alimentar as Arestas do GraphFrames
route_metrics = spark.sql("""
    SELECT
        src,
        dst,
        COUNT(*) AS total_shipments,
        ROUND(SUM(cargo_tons), 2) AS total_tons,
        ROUND(AVG(actual_lead_time_h), 2) AS avg_actual_lead_time_h,
        ROUND(AVG(delay_hours), 2) AS avg_delay_h,
        ROUND(AVG(sla_breached) * 100, 2) AS sla_breach_pct,
        ROUND(SUM(freight_cost_brl), 2) AS total_freight_brl
    FROM silver_shipments
    GROUP BY src, dst
""")

route_metrics.write.mode("overwrite").parquet("data/processed/silver_route_metrics.parquet")

print("\n📊 TOP 5 ROTAS COM MAIOR QUEBRA DE SLA (SPARK SQL):")
spark.sql("""
    SELECT
        src, dst, total_shipments, total_tons,
        avg_actual_lead_time_h, avg_delay_h, sla_breach_pct
    FROM (
        SELECT *, DENSE_RANK() OVER (ORDER BY sla_breach_pct DESC) AS rnk
        FROM (
            SELECT
                src, dst, COUNT(*) AS total_shipments,
                ROUND(SUM(cargo_tons), 1) AS total_tons,
                ROUND(AVG(actual_lead_time_h), 2) AS avg_actual_lead_time_h,
                ROUND(AVG(delay_hours), 2) AS avg_delay_h,
                ROUND(AVG(sla_breached) * 100, 2) AS sla_breach_pct
            FROM silver_shipments
            GROUP BY src, dst
        )
    )
    WHERE rnk <= 5
""").show(truncate=False)

print("🚚 PERFORMANCE POR TRANSPORTADORA (SPARK SQL):")
spark.sql("""
    SELECT
        carrier,
        COUNT(*) AS embarques,
        ROUND(AVG(delay_hours), 2) AS atraso_medio_h,
        ROUND(AVG(sla_breached) * 100, 2) AS quebra_sla_pct,
        ROUND(SUM(freight_cost_brl), 2) AS frete_total_brl
    FROM silver_shipments
    GROUP BY carrier
    ORDER BY quebra_sla_pct DESC
""").show(truncate=False)

spark.stop()
