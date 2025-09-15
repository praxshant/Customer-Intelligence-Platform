#!/bin/bash
set -e

echo "Starting CICOP Development Environment..."

if [ ! -d "venv" ]; then
  python -m venv venv
fi
source venv/bin/activate

pip install -r requirements.txt

python scripts/setup_database.py
python scripts/generate_standalone_data.py

uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload &
UVICORN_PID=$!
uvicorn src.ml.model_server:app --host 0.0.0.0 --port 8001 --reload &
ML_PID=$!
uvicorn src.api.gateway:app --host 0.0.0.0 --port 8002 --reload &
GW_PID=$!

sleep 3
streamlit run src/dashboard/app.py --server.port=8501 --server.address=0.0.0.0 &
DASH_PID=$!

echo "Dashboard: http://localhost:8501"
echo "API Docs: http://localhost:8000/docs"
echo "Gateway: http://localhost:8002/health"
echo "ML: http://localhost:8001/health"

trap 'kill $UVICORN_PID $ML_PID $GW_PID $DASH_PID 2>/dev/null || true; exit 0' SIGINT SIGTERM

wait
