from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import sys
import pandas as pd
import numpy as np
from contextlib import asynccontextmanager
import redis.asyncio as redis
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
import logging
import traceback

# Ensure project root is on the path for absolute imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.database_manager import DatabaseManager
from src.ml.customer_segmentation import CustomerSegmentation
from src.rules.campaign_generator import CampaignGenerator
from src.auth.auth_manager import auth_manager
from src.utils.json_encoder import convert_numpy_types

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize rate limiter
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    try:
        r = redis.from_url(redis_url, encoding="utf-8", decode_responses=True)
        await FastAPILimiter.init(r)
    except Exception:
        # Proceed without limiter if redis not available
        pass
    yield
    # Shutdown
    try:
        await FastAPILimiter.close()
    except Exception:
        pass

# Configure debug logging to stdout
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CICOP API",
    description="Customer Intelligence & Campaign Orchestration Platform API",
    version="1.0.0",
    lifespan=lifespan,
    debug=True,
)

# Security middlewares
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1").split(","),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8501").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"]
)

# Global exception handlers for better diagnostics
@app.exception_handler(500)
async def internal_server_error_handler(request: Request, exc: Exception):
    logger.error(f"Internal server error: {exc}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": str(exc),
            "path": str(request.url),
        },
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    return JSONResponse(status_code=500, content={"error": "Internal Server Error", "detail": str(exc)})

# Global components
_db: Optional[DatabaseManager] = None
_seg: Optional[CustomerSegmentation] = None
_campaigns: Optional[CampaignGenerator] = None

# Rate limiter dependency default
rate_limiter = RateLimiter(times=int(os.getenv("RATE_LIMIT_PER_MINUTE", "100")), seconds=60)


@app.on_event("startup")
async def startup_event():
    global _db, _seg, _campaigns
    try:
        logger.info("Starting up CICOP API components...")
        _db = DatabaseManager()
        _seg = CustomerSegmentation()
        _campaigns = CampaignGenerator()
        logger.info("✅ Components initialized successfully")
    except Exception as e:
        print(f"Error initializing API components with default settings: {e}")
        # Fallback: force SQLite locally and retry
        try:
            os.environ["DATABASE_URL"] = "sqlite:///data/customer_insights.db"
            _db = DatabaseManager()
            _seg = CustomerSegmentation()
            _campaigns = CampaignGenerator()
            print("Initialized API components using SQLite fallback DATABASE_URL.")
        except Exception as e2:
            print(f"Fallback initialization failed: {e2}")


# Pydantic models
class CustomerResponse(BaseModel):
    customer_id: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None
    age: Optional[int] = None
    location: Optional[str] = None


class TransactionResponse(BaseModel):
    transaction_id: str
    customer_id: str
    amount: float
    category: str
    merchant: Optional[str] = None
    transaction_date: str


class CampaignRequest(BaseModel):
    customer_id: str
    campaign_type: Optional[str] = None
    budget: Optional[float] = 1000


class SegmentationRequest(BaseModel):
    n_clusters: Optional[int] = 5


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


# Using convert_numpy_types from src.utils.json_encoder


@app.get("/")
async def root():
    return {
        "message": "CICOP API is running",
        "version": "1.0.0",
        "docs": "/docs",
        "debug": True,
    }


@app.get("/health")
async def health_check():
    try:
        global _db, _seg, _campaigns
        # Lazy init fallback in case startup failed or env changed
        if _db is None:
            try:
                os.environ["DATABASE_URL"] = os.getenv("DATABASE_URL", "sqlite:///data/customer_insights.db") or "sqlite:///data/customer_insights.db"
                _db = DatabaseManager()
                if _seg is None:
                    _seg = CustomerSegmentation()
                if _campaigns is None:
                    _campaigns = CampaignGenerator()
            except Exception as init_err:
                # If init fails, return degraded with error info
                return {
                    "status": "degraded",
                    "database_connected": False,
                    "data_stats": {},
                    "error": f"init_failed: {init_err}"
                }
        # Try to fetch stats, but don't fail hard
        try:
            stats = _db.get_database_stats() if _db else {}
        except Exception as stats_err:
            return {
                "status": "degraded",
                "database_connected": _db is not None,
                "data_stats": {},
                "error": f"stats_failed: {stats_err}"
            }
        return JSONResponse(content=convert_numpy_types({
            "status": "healthy" if _db else "degraded",
            "database_connected": _db is not None,
            "data_stats": stats,
        }))
    except Exception as e:
        # Last resort: never 500 on health
        return JSONResponse(content=convert_numpy_types({
            "status": "degraded",
            "database_connected": False,
            "data_stats": {},
            "error": str(e)
        }), status_code=200)


# Authentication endpoints
@app.post("/auth/login", response_model=TokenResponse)
async def login(login_data: LoginRequest, _: None = Depends(RateLimiter(times=5, seconds=60))):
    user = auth_manager.authenticate_user(login_data.username, login_data.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    access = auth_manager.create_access_token({"sub": user["user_id"]})
    refresh = auth_manager.create_refresh_token({"sub": user["user_id"]})
    return TokenResponse(access_token=access, refresh_token=refresh)


@app.get("/auth/me")
async def whoami(current_user: dict = Depends(auth_manager.get_current_user)):
    return current_user


@app.get("/customers", response_model=List[CustomerResponse])
async def get_customers(limit: int = 100, current_user: dict = Depends(auth_manager.get_current_user), _: None = Depends(rate_limiter)):
    try:
        if not _db:
            raise HTTPException(status_code=500, detail="Database not initialized")
        df = _db.query_to_dataframe(f"SELECT * FROM customers LIMIT {int(limit)}")
        return [] if df.empty else df.to_dict("records")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching customers: {e}")


@app.get("/customers/{customer_id}", response_model=CustomerResponse)
async def get_customer(customer_id: str, current_user: dict = Depends(auth_manager.get_current_user), _: None = Depends(rate_limiter)):
    try:
        if not _db:
            raise HTTPException(status_code=500, detail="Database not initialized")
        df = _db.query_to_dataframe("SELECT * FROM customers WHERE customer_id = ?", (customer_id,))
        if df.empty:
            raise HTTPException(status_code=404, detail="Customer not found")
        return df.iloc[0].to_dict()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching customer: {e}")


@app.get("/customers/{customer_id}/transactions", response_model=List[TransactionResponse])
async def get_customer_transactions(customer_id: str, limit: int = 50, current_user: dict = Depends(auth_manager.get_current_user), _: None = Depends(rate_limiter)):
    try:
        if not _db:
            raise HTTPException(status_code=500, detail="Database not initialized")
        df = _db.query_to_dataframe(
            f"SELECT * FROM transactions WHERE customer_id = ? ORDER BY transaction_date DESC LIMIT {int(limit)}",
            (customer_id,),
        )
        return [] if df.empty else df.to_dict("records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching transactions: {e}")


@app.get("/segments")
async def get_segments(current_user: dict = Depends(auth_manager.get_current_user), _: None = Depends(rate_limiter)):
    try:
        if not _db:
            raise HTTPException(status_code=500, detail="Database not initialized")
        df = _db.query_to_dataframe("SELECT * FROM customer_segments")
        if df.empty:
            return {"segments": [], "summary": {}}
        return {"segments": df.to_dict("records"), "summary": df["segment_name"].value_counts().to_dict()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching segments: {e}")


@app.post("/segmentation")
async def run_segmentation(request: SegmentationRequest, current_user: dict = Depends(auth_manager.get_current_user), _: None = Depends(rate_limiter)):
    try:
        if not _db or not _seg:
            raise HTTPException(status_code=500, detail="Components not initialized")
        tx = _db.query_to_dataframe("SELECT * FROM transactions")
        if tx.empty:
            raise HTTPException(status_code=400, detail="No transaction data available")
        from src.data.data_preprocessor import DataPreprocessor

        pre = DataPreprocessor()
        feats = pre.create_customer_features(tx)
        if feats.empty:
            raise HTTPException(status_code=400, detail="Could not create customer features")
        res = _seg.perform_segmentation(feats, request.n_clusters)
        if res.empty:
            raise HTTPException(status_code=500, detail="Segmentation failed")
        segments_df = res[["customer_id", "segment_id", "segment_name", "last_updated"]].copy()
        _db.insert_dataframe(segments_df, "customer_segments", if_exists="replace")
        return {
            "status": "success",
            "clusters": request.n_clusters,
            "customers_segmented": int(len(res)),
            "segments": res["segment_name"].value_counts().to_dict(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation error: {e}")


@app.post("/campaigns/generate")
async def generate_campaigns(request: CampaignRequest, current_user: dict = Depends(auth_manager.get_current_user), _: None = Depends(rate_limiter)):
    try:
        if not _db or not _campaigns:
            raise HTTPException(status_code=500, detail="Components not initialized")
        cust = _db.query_to_dataframe("SELECT * FROM customers WHERE customer_id = ?", (request.customer_id,))
        if cust.empty:
            raise HTTPException(status_code=404, detail="Customer not found")
        df = _campaigns.generate_personalized_campaigns(cust, cust)
        if df.empty:
            return {"campaigns": [], "message": "No campaigns generated"}
        return {"campaigns": df.to_dict("records"), "total_campaigns": int(len(df))}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Campaign generation error: {e}")


@app.get("/analytics/dashboard")
async def get_dashboard_data(current_user: dict = Depends(auth_manager.get_current_user), _: None = Depends(rate_limiter)):
    try:
        if not _db:
            raise HTTPException(status_code=500, detail="Database not initialized")
        stats = _db.get_database_stats()
        tx = _db.query_to_dataframe("SELECT * FROM transactions LIMIT 10000")
        analytics: Dict[str, Any] = {
            "database_stats": stats,
            "revenue_metrics": {},
            "segment_distribution": {},
            "campaign_metrics": {},
        }
        if not tx.empty:
            analytics["revenue_metrics"] = {
                "total_revenue": float(tx["amount"].sum()),
                "avg_transaction": float(tx["amount"].mean()),
                "transaction_count": int(len(tx)),
                "categories": int(tx["category"].nunique()),
            }
        seg = _db.query_to_dataframe("SELECT * FROM customer_segments")
        if not seg.empty:
            analytics["segment_distribution"] = seg["segment_name"].value_counts().to_dict()
        camp = _db.query_to_dataframe("SELECT * FROM campaigns")
        if not camp.empty:
            analytics["campaign_metrics"] = {
                "total_campaigns": int(len(camp)),
                "active_campaigns": int(len(camp[camp["status"] == "active"])),
                "campaign_types": camp["campaign_type"].value_counts().to_dict(),
            }
        return analytics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analytics error: {e}")


@app.get("/debug")
async def debug_info():
    try:
        import importlib.util
        info: Dict[str, Any] = {
            "python_version": sys.version,
            "working_directory": os.getcwd(),
            "python_path": sys.path[:5],
            "environment_variables": {
                "DATABASE_URL": os.getenv("DATABASE_URL", "Not set"),
                "PYTHONPATH": os.getenv("PYTHONPATH", "Not set"),
            },
            "module_checks": {},
        }
        modules = [
            "src.data.database_manager",
            "src.ml.customer_segmentation",
            "src.rules.rules_engine",
        ]
        for m in modules:
            try:
                spec = importlib.util.find_spec(m)
                info["module_checks"][m] = "✅ Found" if spec else "❌ Not found"
            except Exception as me:
                info["module_checks"][m] = f"❌ Error: {me}"
        return convert_numpy_types(info)
    except Exception as e:
        logger.error(f"/debug failed: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server...")
    logger.info(f"Database URL: {os.getenv('DATABASE_URL', 'Not set')}")
    logger.info(f"Python path: {sys.path[:5]}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
