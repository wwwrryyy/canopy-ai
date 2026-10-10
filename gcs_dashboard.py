"""
gcs_dashboard.py
================
Ground Control Station (GCS) Analytics Dashboard for Aerial Precision Agriculture.
Connects to the drone's SQLite offline telemetry spooler and visualizes
detected anomalies and variable-rate application maps on an interactive dashboard.
"""

import os
import time
import json
import sqlite3
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

# Streamlit Page Config
st.set_page_config(page_title="Canopy AI GCS Dashboard", layout="wide", page_icon="🚁")

# Apply Apple Design Principles via CSS injection
apple_design_css = """
<style>
    /* Typography: system font, tight tracking on headers */
    :root { font: 100%/1.5 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important; }
    h1, h2, h3, h4, h5, h6 { font-optical-sizing: auto; font-weight: 600 !important; }
    h1 { letter-spacing: -0.02em !important; line-height: 1.05 !important; }
    h2 { letter-spacing: -0.01em !important; line-height: 1.1 !important; }
    
    /* Materials & Depth: Translucency conveying hierarchy */
    [data-testid="stHeader"] {
        background: rgba(255, 255, 255, 0.6) !important;
        backdrop-filter: blur(20px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
        border-bottom: 1px solid rgba(0,0,0,0.05) !important;
    }
    [data-testid="stSidebar"] {
        background: rgba(248, 249, 250, 0.65) !important;
        backdrop-filter: blur(20px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
        border-right: 1px solid rgba(0,0,0,0.05) !important;
    }
    
    /* Response: Instant feedback on interaction */
    .stButton > button, .stCheckbox > div {
        transition: transform 100ms ease-out !important;
    }
    .stButton > button:active, .stCheckbox > div:active {
        transform: scale(0.97) !important;
    }
    
    /* Metrics: Soft shadows and subtle hover states */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.8);
        border-radius: 16px;
        padding: 16px 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
        transition: transform 200ms cubic-bezier(0.2, 0.8, 0.2, 1), box-shadow 200ms ease-out !important;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px) scale(1.01) !important;
        box-shadow: 0 8px 24px rgba(0,0,0,0.06);
    }
    [data-testid="stMetricValue"] {
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
        color: #1d1d1f !important;
    }
    [data-testid="stMetricLabel"] {
        font-weight: 500 !important;
        color: #86868b !important;
    }
</style>
"""
st.markdown(apple_design_css, unsafe_allow_html=True)

st.title("🚁 Canopy AI: Precision Agriculture GCS")
st.markdown("Ground Control Station dashboard for reviewing drone anomaly telemetry and Variable Rate Application (VRA) maps.")

# Paths
DB_PATH = "exports/flight_telemetry.db"
GEOJSON_PATH = "exports/prescription_map.geojson"

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
@st.cache_data(ttl=5)
def load_telemetry_data(db_path: str) -> pd.DataFrame:
    """Loads anomaly logs from the SQLite spooler."""
    if not os.path.exists(db_path):
        return pd.DataFrame()
        
    try:
        conn = sqlite3.connect(db_path)
        # Select relevant columns, excluding raw bbox_json for the UI overview
        query = """
            SELECT timestamp, anomaly_id, anomaly_type, confidence, mse_score, 
                   latitude, longitude, altitude, estimated_area_sqm
            FROM anomaly_logs
            ORDER BY timestamp DESC
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if not df.empty:
            # Convert Unix timestamp to datetime
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            
        return df
    except Exception as e:
        st.error(f"Error loading database: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=5)
def load_geojson_map(geojson_path: str) -> dict:
    """Loads the VRA prescription map GeoJSON."""
    if not os.path.exists(geojson_path):
        return {}
    try:
        with open(geojson_path, "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading GeoJSON: {e}")
        return {}

df_logs = load_telemetry_data(DB_PATH)
geojson_data = load_geojson_map(GEOJSON_PATH)

# ---------------------------------------------------------------------------
# Sidebar Filters
# ---------------------------------------------------------------------------
st.sidebar.header("Filters")
if not df_logs.empty:
    unique_types = ["All"] + list(df_logs["anomaly_type"].unique())
    selected_type = st.sidebar.selectbox("Filter by Pathology", unique_types)
    
    min_conf = st.sidebar.slider("Minimum Confidence", 0.0, 1.0, 0.5)
    
    # Apply filters
    if selected_type != "All":
        df_filtered = df_logs[df_logs["anomaly_type"] == selected_type]
    else:
        df_filtered = df_logs
        
    df_filtered = df_filtered[df_filtered["confidence"] >= min_conf]
else:
    df_filtered = pd.DataFrame()
    st.sidebar.warning("No telemetry data found. Is the drone database synced?")

st.sidebar.markdown("---")
st.sidebar.header("Live Edge Feed")
auto_refresh = st.sidebar.checkbox("Enable Auto-Refresh", value=False)
LATEST_FRAME_PATH = "exports/latest_frame.jpg"
if os.path.exists(LATEST_FRAME_PATH):
    st.sidebar.image(LATEST_FRAME_PATH, caption="Latest Edge Detection", use_column_width=True)
else:
    st.sidebar.info("Waiting for edge node feed...")



# ---------------------------------------------------------------------------
# KPI Metrics
# ---------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

total_anomalies = len(df_filtered)
high_conf = len(df_filtered[df_filtered["confidence"] > 0.85]) if not df_filtered.empty else 0
total_area = df_filtered["estimated_area_sqm"].sum() if not df_filtered.empty else 0.0

# Calculate total liters based on GeoJSON if available
total_liters = 0.0
if geojson_data and "features" in geojson_data:
    for feature in geojson_data["features"]:
        rate = feature.get("properties", {}).get("application_rate_L_ha", 0.0)
        area_sqm = feature.get("properties", {}).get("area_sqm", 0.0)
        # L/ha * (ha) = L (1 ha = 10,000 sqm)
        total_liters += rate * (area_sqm / 10000.0)

col1.metric("Detected Anomalies", total_anomalies)
col2.metric("High Confidence (>85%)", high_conf)
col3.metric("Affected Area (sqm)", f"{total_area:.2f}")
col4.metric("Treatment Required (Liters)", f"{total_liters:.4f}")

st.markdown("---")

# ---------------------------------------------------------------------------
# Interactive Map
# ---------------------------------------------------------------------------
st.subheader("Geospatial Anomaly Map")

if not df_filtered.empty and not df_filtered["latitude"].isna().all():
    # Center map on the median coordinates
    center_lat = df_filtered["latitude"].dropna().median()
    center_lon = df_filtered["longitude"].dropna().median()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=18, tiles="CartoDB positron")
    
    # Plot GeoJSON Prescription Map if available
    if geojson_data and "features" in geojson_data:
        def style_function(feature):
            pathology = feature["properties"].get("target_type", "")
            color = "#3388ff" # Default blue
            if "Rust" in pathology or "Fungal" in pathology:
                color = "#e31a1c" # Red
            elif "Pest" in pathology:
                color = "#ff7f00" # Orange
            elif "Water" in pathology or "Drought" in pathology:
                color = "#ffff33" # Yellow
            elif "Weed" in pathology:
                color = "#33a02c" # Green
                
            return {
                "fillColor": color,
                "color": "black",
                "weight": 1,
                "fillOpacity": 0.6,
            }
            
        folium.GeoJson(
            geojson_data,
            style_function=style_function,
            tooltip=folium.GeoJsonTooltip(
                fields=["id", "target_type", "application_rate_L_ha", "area_sqm"],
                aliases=["Anomaly ID:", "Pathology:", "Treatment Rate (L/ha):", "Area (sqm):"]
            )
        ).add_to(m)
        
    # Plot individual point markers for the filtered anomalies
    for idx, row in df_filtered.iterrows():
        if pd.isna(row["latitude"]) or pd.isna(row["longitude"]):
            continue
            
        popup_html = f"""
        <b>Type:</b> {row['anomaly_type']}<br>
        <b>Confidence:</b> {row['confidence']:.2f}<br>
        <b>MSE Score:</b> {row['mse_score']:.4f}<br>
        """
        
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=3,
            color="black",
            fill=True,
            fill_color="black",
            popup=folium.Popup(popup_html, max_width=200)
        ).add_to(m)
        
    # Render Folium map in Streamlit
    st_folium(m, width=1200, height=600)
else:
    st.info("No geospatial data available to display on the map.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Data Table
# ---------------------------------------------------------------------------
st.subheader("Raw Telemetry Feed")
if not df_filtered.empty:
    display_cols = ["datetime", "anomaly_id", "anomaly_type", "confidence", "mse_score", "latitude", "longitude", "altitude"]
    st.dataframe(df_filtered[display_cols].style.format({
        "confidence": "{:.2f}",
        "mse_score": "{:.4f}",
        "latitude": "{:.6f}",
        "longitude": "{:.6f}",
        "altitude": "{:.1f}"
    }), use_container_width=True)
else:
    st.info("No telemetry logs to display.")


# Perform auto-refresh after all page content has rendered
if auto_refresh:
    time.sleep(1)
    st.rerun()
