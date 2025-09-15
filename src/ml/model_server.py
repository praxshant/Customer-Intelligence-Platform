# src/ml/model_server.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np
import os
import joblib
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

app = FastAPI(title="CICOP ML Model Server", version="0.2.0")


# Prometheus metrics
PREDICT_REQUESTS = Counter("ml_predict_requests_total", "Total number of predict requests")
PREDICT_ERRORS = Counter("ml_predict_errors_total", "Total number of predict errors")
PREDICT_LATENCY = Histogram("ml_predict_latency_seconds", "Latency of predict requests")


class PredictRequest(BaseModel):
    features: List[float]
    metadata: Optional[Dict[str, Any]] = None


class PredictBatchRequest(BaseModel):
    batch: List[List[float]]
    metadata: Optional[Dict[str, Any]] = None


MODEL_PATH = os.getenv("MODEL_PATH", os.path.join("models", "model.pkl"))
_model = None


def _load_model() -> Optional[object]:
    global _model
    if os.path.exists(MODEL_PATH):
        _model = joblib.load(MODEL_PATH)
        return _model
    _model = None
    return None


@app.on_event("startup")
def startup_event():
    _load_model()


@app.get("/health")
async def health():
    return {"status": "ok", "service": "ml-serving", "model_loaded": _model is not None}


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/reload_model")
async def reload_model():
    try:
        m = _load_model()
        return {"reloaded": bool(m), "path": MODEL_PATH}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
async def predict(req: PredictRequest):
    PREDICT_REQUESTS.inc()
    with PREDICT_LATENCY.time():
        try:
            x = np.array(req.features, dtype=float).reshape(1, -1)
            if _model is not None and hasattr(_model, "predict_proba"):
                proba = float(_model.predict_proba(x)[0, 1])
                return {"score": proba, "label": int(proba > 0.5)}
            elif _model is not None and hasattr(_model, "predict"):
                pred = int(_model.predict(x)[0])
                return {"score": float(pred), "label": pred}
            else:
                # Fallback placeholder if no model available
                s = float(1 / (1 + np.exp(-x.sum() / (x.shape[1] or 1))))
                return {"score": s, "label": int(s > 0.5), "placeholder": True}
        except Exception as e:
            PREDICT_ERRORS.inc()
            raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict_batch")
async def predict_batch(req: PredictBatchRequest):
    PREDICT_REQUESTS.inc()
    with PREDICT_LATENCY.time():
        try:
            scores = []
            labels = []
            for row in req.batch:
                x = np.array(row, dtype=float).reshape(1, -1)
                if _model is not None and hasattr(_model, "predict_proba"):
                    proba = float(_model.predict_proba(x)[0, 1])
                    scores.append(proba)
                    labels.append(int(proba > 0.5))
                elif _model is not None and hasattr(_model, "predict"):
                    pred = int(_model.predict(x)[0])
                    scores.append(float(pred))
                    labels.append(pred)
                else:
                    s = float(1 / (1 + np.exp(-x.sum() / (x.shape[1] or 1))))
                    scores.append(s)
                    labels.append(int(s > 0.5))
            return {"scores": scores, "labels": labels}
        except Exception as e:
            PREDICT_ERRORS.inc()
            raise HTTPException(status_code=400, detail=str(e))
