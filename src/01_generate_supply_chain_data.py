import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)

nodes_data = [
    ("SUP_01", "AgroInsumos Paraná", "SUPPLIER", "PR", 5000),
    ("SUP_02", "Química Sul Brasil", "SUPPLIER", "RS", 4500),
    ("SUP_03", "Embalagens Minas", "SUPPLIER", "MG", 4000),
    ("SUP_04", "Polímeros Bahia", "SUPPLIER", "BA", 3800),
    ("SUP_05", "Metais Goiás", "SUPPLIER", "GO", 4200),
    ("FAC_01", "Planta Industrial Jundiaí", "FACTORY", "SP", 9000),
    ("FAC_02", "Planta Industrial Curitiba", "FACTORY", "PR", 7500),
    ("FAC_03", "Planta Industrial Betim", "FACTORY", "MG", 7000),
    ("DC_01", "CD MegaHub Campinas", "DC", "SP", 15000),
    ("DC_02", "CD Regional Sorocaba", "DC", "SP", 8000),
    ("DC_03", "CD Regional Goiânia", "DC", "GO", 7500),
    ("DC_04", "CD Nordeste Recife", "DC", "PE", 6500),
    ("STR_01", "Loja Flagship Paulista", "STORE", "SP", 2000),
    ("STR_02", "Loja Campinas Centro", "STORE", "SP", 1800),
    ("STR_03", "Loja Sorocaba Norte", "STORE", "SP", 1600),
    ("STR_04", "Loja Brasília Asa Sul", "STORE", "DF", 1900),
    ("STR_05", "Loja Goiânia Bueno", "STORE", "GO", 1500),
    ("STR_06", "Loja Belo Horizonte Savassi", "STORE", "MG", 1700),
    ("STR_07", "Loja Curitiba Batel", "STORE", "PR", 1600),
    ("STR_08", "Loja Recife Boa Viagem", "STORE", "PE", 1500),
    ("STR_09", "Loja Salvador Pituba", "STORE", "BA", 1400),
    ("STR_10", "Loja Rio Barra", "STORE", "RJ", 2100),
]

nodes_df = pd.DataFrame(nodes_data, columns=["id", "name", "node_type", "state", "capacity_tons"])

routes_data = [
    ("SUP_01", "FAC_02", 180, 4.0, 85.0),
    ("SUP_01", "FAC_01", 460, 8.5, 160.0),
    ("SUP_02", "FAC_02", 710, 12.0, 230.0),
    ("SUP_03", "FAC_03", 90, 2.5, 60.0),
    ("SUP_03", "FAC_01", 560, 9.5, 185.0),
    ("SUP_04", "FAC_03", 1350, 22.0, 410.0),
    ("SUP_05", "FAC_03", 890, 15.0, 290.0),
    ("SUP_05", "FAC_01", 920, 15.5, 300.0),
    ("FAC_01", "DC_01", 45, 1.5, 45.0),
    ("FAC_01", "DC_02", 95, 2.5, 65.0),
    ("FAC_02", "DC_01", 490, 8.0, 175.0),
    ("FAC_02", "DC_02", 410, 7.0, 150.0),
    ("FAC_03", "DC_01", 590, 9.5, 195.0),
    ("FAC_03", "DC_03", 730, 12.0, 240.0),
    ("FAC_03", "DC_04", 2050, 32.0, 620.0),
    ("DC_01", "DC_02", 90, 2.0, 55.0),
    ("DC_01", "DC_03", 840, 13.5, 270.0),
    ("DC_02", "DC_01", 90, 2.0, 55.0),
    ("DC_01", "STR_01", 95, 2.5, 70.0),
    ("DC_01", "STR_02", 18, 1.0, 35.0),
    ("DC_01", "STR_06", 580, 9.0, 190.0),
    ("DC_01", "STR_07", 495, 8.0, 170.0),
    ("DC_01", "STR_10", 510, 8.5, 180.0),
    ("DC_02", "STR_01", 105, 2.5, 75.0),
    ("DC_02", "STR_03", 15, 1.0, 30.0),
    ("DC_02", "STR_07", 405, 7.0, 145.0),
    ("DC_03", "STR_04", 210, 3.5, 95.0),
    ("DC_03", "STR_05", 20, 1.0, 35.0),
    ("DC_04", "STR_08", 25, 1.0, 40.0),
    ("DC_04", "STR_09", 805, 13.0, 260.0),
]

routes_df = pd.DataFrame(
    routes_data,
    columns=["src", "dst", "distance_km", "std_lead_time_h", "base_cost_per_ton"]
)

num_shipments = 15000
carriers = ["TransLog Rápido", "Rodoviário Brasil", "Expresso Mercosul", "CargoSul"]
carrier_penalty = {
    "TransLog Rápido": -0.8,
    "Rodoviário Brasil": 0.4,
    "Expresso Mercosul": 0.0,
    "CargoSul": 2.2,
}

start_date = datetime(2025, 1, 1)
route_indices = np.random.randint(0, len(routes_df), size=num_shipments)
selected_routes = routes_df.iloc[route_indices].reset_index(drop=True)

days_offset = np.random.randint(0, 650, size=num_shipments)
dispatch_dates = [start_date + timedelta(days=int(d)) for d in days_offset]
months = np.array([dt.month for dt in dispatch_dates])
day_of_week = np.array([dt.weekday() for dt in dispatch_dates])

cargo_tons = np.round(np.random.uniform(2.0, 32.0, size=num_shipments), 2)
selected_carriers = np.random.choice(carriers, size=num_shipments, p=[0.30, 0.30, 0.25, 0.15])
weather_severity = np.random.choice([0, 1, 2], size=num_shipments, p=[0.70, 0.22, 0.08])

is_peak_season = np.isin(months, [11, 12]).astype(float)
is_hub_campinas = ((selected_routes["dst"] == "DC_01") | (selected_routes["src"] == "DC_01")).astype(float)
carrier_effect = np.array([carrier_penalty[c] for c in selected_carriers])

noise = np.random.normal(0, 0.9, size=num_shipments)
delay_hours = (
    0.002 * selected_routes["distance_km"].values
    + 0.06 * cargo_tons
    + 1.9 * weather_severity
    + 2.4 * is_peak_season
    + 1.6 * is_hub_campinas
    + carrier_effect
    - 1.8
    + noise
)
delay_hours = np.round(np.maximum(0.0, delay_hours), 2)
actual_lead_time_h = np.round(selected_routes["std_lead_time_h"].values + delay_hours, 2)
sla_hours = np.round(selected_routes["std_lead_time_h"].values * 1.25 + 1.5, 2)
freight_cost = np.round(
    selected_routes["base_cost_per_ton"].values * cargo_tons * (1 + 0.08 * is_peak_season), 2
)

shipments_df = pd.DataFrame({
    "shipment_id": [f"SHP_{i:06d}" for i in range(1, num_shipments + 1)],
    "dispatch_date": [dt.strftime("%Y-%m-%d") for dt in dispatch_dates],
    "month": months,
    "day_of_week": day_of_week,
    "src": selected_routes["src"],
    "dst": selected_routes["dst"],
    "carrier": selected_carriers,
    "cargo_tons": cargo_tons,
    "weather_severity": weather_severity,
    "distance_km": selected_routes["distance_km"],
    "std_lead_time_h": selected_routes["std_lead_time_h"],
    "sla_hours": sla_hours,
    "actual_lead_time_h": actual_lead_time_h,
    "delay_hours": delay_hours,
    "sla_breached": (actual_lead_time_h > sla_hours).astype(int),
    "freight_cost_brl": freight_cost,
})

nodes_df.to_parquet(f"{RAW_DIR}/nodes.parquet", index=False)
routes_df.to_parquet(f"{RAW_DIR}/routes.parquet", index=False)
shipments_df.to_parquet(f"{RAW_DIR}/shipments.parquet", index=False)

print(f"✅ Bronze atualizado | Taxa realista de quebra de SLA: {shipments_df['sla_breached'].mean()*100:.2f}%")
