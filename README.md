## 🚚 Supply Chain Graph Lakehouse & Predictive Logistics



![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PySpark](https://img.shields.io/badge/PySpark-3.5.3-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white)
![Spark SQL](https://img.shields.io/badge/Spark_SQL-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white)
![GraphFrames](https://img.shields.io/badge/GraphFrames-0.8.3-2E7D32?style=for-the-badge&logo=apachespark&logoColor=white)
![Spark MLlib](https://img.shields.io/badge/Spark_MLlib-1565C0?style=for-the-badge&logo=apachespark&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey.svg?style=for-the-badge)](https://creativecommons.org/licenses/by-nc/4.0/)



Pipeline end-to-end de **Engenharia de Dados, Análise de Grafos e Machine Learning Distribuído** construído com **PySpark, Spark SQL, GraphFrames e Spark MLlib** em ambiente containerizado (**Docker + OpenJDK 17**), acompanhado de dashboard interativo em **Streamlit**.



O projeto modela uma malha logística multi-tier (`Fornecedor -> Fábrica -> Centro de Distribuição -> Loja`), identifica gargalos estruturais e pontos únicos de falha na rede de suprimentos e utiliza métricas topológicas extraídas do grafo para alimentar modelos preditivos de atraso e quebra de SLA.



 ## 🏗️ Arquitetura Medallion & Stack Tecnológica

- **Infraestrutura & Isolamento:** Docker & Docker Compose (Python 3.11 + OpenJDK 17 + Cache Maven persistente).

- **Camada Bronze (`src/01_generate_supply_chain_data.py`):** Geração determinística de 22 nós operacionais, 30 rotas direcionadas e 15.000 embarques históricos em armazenamento colunar **Parquet**.

- **Camada Silver (`src/02_spark_sql_silver_analytics.py`):** Transformações distribuídas via **Spark SQL**, joins relacionais, cálculo de custos unitários e *Window Functions* (`AVG OVER ROWS BETWEEN 30 PRECEDING`) para monitoramento de atraso móvel por rota.

- **Camada Gold — Topologia (`src/03_graphframes_topology.py`):** Modelagem de grafo direcionado com **GraphFrames** aplicando *PageRank*, *In/Out-Degree*, *Shortest Paths*, *Motif Finding* e simulação de falha crítica via *Connected Components*.

- **Camada Gold — Machine Learning (`src/04_mllib_delay_prediction.py`):** Feature engineering unindo atributos operacionais e métricas de centralidade do grafo em pipelines nativos do **Spark MLlib** (`GBTRegressor` + `RandomForestClassifier`).

- **Visualização Interativa (`app.py`):** Aplicação web em **Streamlit**, **Plotly** e **NetworkX** para simulação visual de contingência e auditoria das predições.


## 📊 Principais Descobertas & Resultados Obtidos

1. **Identificação de Gargalo Sistêmico (GraphFrames PageRank):**

   * O hub **DC_01 (CD MegaHub Campinas)** concentrou o maior score de centralidade da malha (`PageRank: 2.2916`, `in_degree: 4`, `out_degree: 7`), respondendo pelas 5 rotas com maior taxa de quebra de SLA (entre **46,1% e 53,3%**).

2. **Simulação de Contingência & Resiliência (Connected Components):**

   * A simulação de indisponibilidade do **DC_01** fragmentou a malha em **4 componentes desconexos**: enquanto o **DC_02 (Sorocaba)** absorveu 18 nós mantendo o fluxo principal ativo, **3 lojas regionais (`STR_02`, `STR_06` e `STR_10`) ficaram 100% ilhadas**, evidenciando necessidade de rotas de redundância.

3. **Performance Preditiva (Spark MLlib):**

   * **Regressão de Atraso (`GBTRegressor`):** **RMSE de 0.841 horas** e **$R^2$ de 0.8167**.

   * **Classificação de Risco de SLA (`RandomForestClassifier`):** **AUC-ROC de 0.9381**, comprovando o ganho preditivo das features topológicas (`src_pagerank`, `src_out_degree` e `dst_pagerank`) combinadas com severidade climática e sazonalidade.


## ⚡ Como Executar

# 1. Subir o ambiente Spark + GraphFrames via Docker Compose

docker compose up -d --build

# 2. Executar o pipeline completo (Bronze -> Silver -> GraphFrames -> MLlib)

./run_pipeline.sh


# 3. Iniciar o dashboard interativo na porta 8501

docker exec -d spark-supply-chain streamlit run app.py --server.port=8501 --server.address=0.0.0.0


Acesse **`http://localhost:8501`** no navegador para explorar o simulador de topologia e os resultados dos modelos.


## 📄 Licença

Este projeto está licenciado sob a **[Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](LICENSE)** — você pode compartilhar e adaptar o código livremente mediante atribuição de crédito, **vedado o uso para fins comerciais**.