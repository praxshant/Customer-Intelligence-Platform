# src/api/gateway.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import httpx
from fastapi.responses import Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

app = FastAPI(title="CICOP API Gateway", version="0.1.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ML_SERVING_URL = os.getenv("ML_SERVING_URL", "http://localhost:8001")

# Metrics
GW_REQUESTS = Counter("gateway_requests_total", "Total number of gateway requests", ["route"]) 
GW_ERRORS = Counter("gateway_errors_total", "Total number of gateway errors", ["route"]) 
GW_LATENCY = Histogram("gateway_request_latency_seconds", "Latency of gateway requests", ["route"]) 

@app.get("/health")
async def health():
    return {"status": "ok", "service": "api-gateway"}

@app.get("/ml/health")
async def ml_health():
    route = "/ml/health"
    GW_REQUESTS.labels(route).inc()
    with GW_LATENCY.labels(route).time():
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{ML_SERVING_URL}/health")
                return {"upstream_status": r.json()}
        except Exception as e:
            GW_ERRORS.labels(route).inc()
            raise HTTPException(status_code=502, detail=f"ML service unreachable: {e}")

@app.post("/predict")
async def predict(payload: dict):
    route = "/predict"
    GW_REQUESTS.labels(route).inc()
    with GW_LATENCY.labels(route).time():
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.post(f"{ML_SERVING_URL}/predict", json=payload)
                r.raise_for_status()
                return r.json()
        except httpx.HTTPStatusError as e:
            GW_ERRORS.labels(route).inc()
            raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
        except Exception as e:
            GW_ERRORS.labels(route).inc()
            raise HTTPException(status_code=502, detail=f"Upstream error: {e}")

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
