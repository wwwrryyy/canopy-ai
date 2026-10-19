#!/bin/bash
set -e

echo "=========================================="
echo " Canopy AI Real-Time Demonstration"
echo "=========================================="

echo "[1/3] Clearing old telemetry and exports..."
rm -f exports/flight_telemetry.db
rm -f exports/prescription_map.geojson
rm -f exports/latest_frame.jpg

echo "[2/3] Starting GCS Dashboard in the background..."
streamlit run gcs_dashboard.py --server.port 8501 > streamlit.log 2>&1 &
STREAMLIT_PID=$!

cleanup() {
    kill "$STREAMLIT_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Dashboard starting on http://localhost:8501"
echo "Please open the dashboard in your browser and enable the 'Auto-Refresh' toggle in the sidebar."
echo "Waiting 3 seconds for dashboard to initialize..."
sleep 3

echo "[3/3] Starting Edge Node Inference Simulator..."
# Run the edge node at 2 FPS to allow you to watch the live feed updates
python deployment/edge_node.py --fps 2.0 --input-stream data/ucmerced_prepared/agricultural_patches

echo "Edge node mission complete."
echo "Cleaning up..."
cleanup
trap - EXIT INT TERM
echo "Done."
