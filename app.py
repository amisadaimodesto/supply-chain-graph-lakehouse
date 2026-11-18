import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Supply Chain Graph Lakehouse", page_icon="🚚", layout="wide")

@st.cache_data
def load_data():
    nodes = pd.read_parquet("data/raw/nodes.parquet")
    routes = pd.read_parquet("data/raw/routes.parquet")
    silver_shipments = pd.read_parquet("data/processed/silver_shipments.parquet")
    route_metrics = pd.read_parquet("data/processed/silver_route_metrics.parquet")
    centrality = pd.read_parquet("data/processed/gold_node_centrality.parquet")
    predictions = pd.read_parquet("data/processed/gold_shipment_predictions.parquet")
    return nodes, routes, silver_shipments, route_metrics, centrality, predictions

nodes, routes, silver_shipments, route_metrics, centrality, predictions = load_data()

st.title("🚚 Supply Chain Graph Lakehouse & Predictive Logistics")
st.caption("Arquitetura Medallion (Bronze/Silver/Gold) | PySpark • Spark SQL • GraphFrames • Spark MLlib")

# KPIs Executivos no Topo
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Embarques Analisados", f"{len(silver_shipments):,}".replace(",", "."))
col2.metric("Volume Total (Ton)", f"{silver_shipments['cargo_tons'].sum():,.0f}".replace(",", "."))
col3.metric("Taxa Global Quebra SLA", f"{silver_shipments['sla_breached'].mean()*100:.2f}%")
col4.metric("Hub Crítico (PageRank)", "DC_01 (2.2916)")
col5.metric("AUC-ROC Modelo MLlib", "0.9381")

tab1, tab2, tab3 = st.tabs([
    "🕸️ Topologia & Contingência (GraphFrames)",
    "📊 Performance & SLA (Spark SQL)",
    "🤖 Predições de Atraso (Spark MLlib)"
])

with tab1:
    st.subheader("Simulador de Resiliência da Malha Logística")
    c_left, c_right = st.columns([1, 3])
    with c_left:
        offline_nodes = st.multiselect(
            "Simular falha / indisponibilidade de nós:",
            options=nodes["id"].tolist(),
            default=["DC_01"]
        )
        st.info("Remova ou restaure nós acima para visualizar o impacto em tempo real nos componentes conectados da rede.")

        # Calcular componentes isolados após remoção
        G_sim = nx.DiGraph()
        active_nodes = nodes[~nodes["id"].isin(offline_nodes)]
        for _, r in active_nodes.iterrows():
            G_sim.add_node(r["id"])
        active_routes = routes[(~routes["src"].isin(offline_nodes)) & (~routes["dst"].isin(offline_nodes))]
        for _, r in active_routes.iterrows():
            G_sim.add_edge(r["src"], r["dst"])

        components = list(nx.weakly_connected_components(G_sim))
        isolated_stores = [
            list(comp)[0] for comp in components
            if len(comp) == 1 and list(comp)[0].startswith("STR_")
        ]
        st.metric("Componentes Desconexos", len(components))
        if isolated_stores:
            st.error(f"Lojas 100% ilhadas: {', '.join(sorted(isolated_stores))}")
        else:
            st.success("Nenhuma loja isolada na topologia atual.")

    with c_right:
        # Layout hierárquico por Tier
        tier_x = {"SUPPLIER": 0, "FACTORY": 1, "DC": 2, "STORE": 3}
        nodes_merged = nodes.merge(centrality[["id", "pagerank_score", "in_degree", "out_degree"]], on="id", how="left")
        nodes_merged["x"] = nodes_merged["node_type"].map(tier_x)
        nodes_merged["y"] = nodes_merged.groupby("node_type").cumcount()
        nodes_merged["y"] = nodes_merged.groupby("node_type")["y"].transform(lambda s: s - s.mean())
        pos = {r["id"]: (r["x"], r["y"]) for _, r in nodes_merged.iterrows()}

        edge_x, edge_y = [], []
        for _, r in active_routes.iterrows():
            x0, y0 = pos[r["src"]]
            x1, y1 = pos[r["dst"]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        fig_net = go.Figure()
        fig_net.add_trace(go.Scatter(
            x=edge_x, y=edge_y, mode="lines",
            line=dict(width=1.5, color="#888888"), hoverinfo="none", name="Rotas Ativas"
        ))

        active_df = nodes_merged[~nodes_merged["id"].isin(offline_nodes)]
        fig_net.add_trace(go.Scatter(
            x=active_df["x"], y=active_df["y"],
            mode="markers+text",
            text=active_df["id"],
            textposition="top center",
            marker=dict(
                size=active_df["pagerank_score"] * 18,
                color=active_df["pagerank_score"],
                colorscale="YlOrRd",
                showscale=True,
                colorbar=dict(title="PageRank"),
                line=dict(width=1.5, color="black")
            ),
            hovertext=active_df.apply(
                lambda r: f"{r['id']} - {r['name']}<br>Tipo: {r['node_type']} ({r['state']})<br>PageRank: {r['pagerank_score']}<br>In: {r['in_degree']} | Out: {r['out_degree']}",
                axis=1
            ),
            hoverinfo="text",
            name="Nós Ativos"
        ))
        fig_net.update_layout(
            height=500,
            xaxis=dict(tickvals=[0, 1, 2, 3], ticktext=["Tier 1: Fornecedores", "Tier 2: Fábricas", "Tier 3: CDs", "Tier 4: Lojas"], showgrid=False, zeroline=False),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=False
        )
        st.plotly_chart(fig_net, use_container_width=True)

    st.dataframe(centrality, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Análise Operacional por Rota e Transportadora (Camada Silver)")
    c1, c2 = st.columns(2)
    with c1:
        top_routes = route_metrics.sort_values("sla_breach_pct", ascending=False).head(10).copy()
        top_routes["rota"] = top_routes["src"] + " → " + top_routes["dst"]
        fig_r = px.bar(
            top_routes, x="sla_breach_pct", y="rota", orientation="h",
            color="avg_delay_h", color_continuous_scale="Reds",
            title="Top 10 Rotas com Maior Quebra de SLA (%)",
            labels={"sla_breach_pct": "Quebra de SLA (%)", "rota": "Rota", "avg_delay_h": "Atraso Médio (h)"}
        )
        fig_r.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_r, use_container_width=True)

    with c2:
        carrier_perf = silver_shipments.groupby("carrier", as_index=False).agg(
            embarques=("shipment_id", "count"),
            atraso_medio_h=("delay_hours", "mean"),
            quebra_sla_pct=("sla_breached", lambda s: s.mean() * 100),
            frete_total=("freight_cost_brl", "sum")
        )
        fig_c = px.bar(
            carrier_perf.sort_values("quebra_sla_pct", ascending=False),
            x="carrier", y="quebra_sla_pct", color="atraso_medio_h",
            color_continuous_scale="OrRd",
            title="Quebra de SLA (%) por Transportadora",
            labels={"carrier": "Transportadora", "quebra_sla_pct": "Quebra de SLA (%)", "atraso_medio_h": "Atraso Médio (h)"}
        )
        st.plotly_chart(fig_c, use_container_width=True)

with tab3:
    st.subheader("Predição de Atraso (GBTRegressor) & Risco de SLA (RandomForest)")
    c1, c2 = st.columns([2, 1])
    with c1:
        sample_preds = predictions.sample(min(1500, len(predictions)), random_state=42)
        fig_p = px.scatter(
            sample_preds, x="delay_hours", y="predicted_delay_h",
            color=sample_preds["sla_breached"].map({1: "Quebrou SLA", 0: "No Prazo"}),
            opacity=0.65,
            title="Atraso Real vs. Previsto pelo Spark MLlib (R² = 0.8167 | RMSE = 0.841h)",
            labels={"delay_hours": "Atraso Real (h)", "predicted_delay_h": "Atraso Previsto (h)", "color": "Status SLA"}
        )
        st.plotly_chart(fig_p, use_container_width=True)
    with c2:
        min_pred = st.slider("Filtrar atraso previsto mínimo (horas):", 0.0, 10.0, 5.0, 0.5)
        high_risk = predictions[predictions["predicted_delay_h"] >= min_pred].sort_values("predicted_delay_h", ascending=False)
        st.metric("Embarques Críticos Filtrados", len(high_risk))
        st.dataframe(
            high_risk[["shipment_id", "src", "dst", "carrier", "predicted_delay_h", "delay_hours"]].head(15),
            use_container_width=True, hide_index=True
        )
