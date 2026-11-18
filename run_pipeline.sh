
#!/usr/bin/env bash

set -e



echo "🚀 [1/4] Gerando malha logística sintética (Camada Bronze)..."

docker exec spark-supply-chain python3 src/01_generate_supply_chain_data.py



echo "⚙️ [2/4] Processando KPIs e Window Functions via Spark SQL (Camada Silver)..."

docker exec spark-supply-chain python3 src/02_spark_sql_silver_analytics.py



echo "🕸️ [3/4] Executando análise de centralidade e contingência via GraphFrames (Camada Gold)..."

docker exec spark-supply-chain python3 src/03_graphframes_topology.py



echo "🤖 [4/4] Treinando modelos preditivos com Spark MLlib (Camada Gold)..."

docker exec spark-supply-chain python3 src/04_mllib_delay_prediction.py



echo "✅ Pipeline Supply Chain concluído com sucesso!"

