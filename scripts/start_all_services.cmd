@echo off
setlocal ENABLEDELAYEDEXPANSION

REM ======================================================
REM Customer Intelligence Platform - Start All Services
REM - API (FastAPI/Uvicorn)        : http://localhost:8000
REM - ML Model Server (Uvicorn)    : http://localhost:8001
REM - API Gateway (Uvicorn)        : http://localhost:8002
REM - Streamlit Dashboard          : http://localhost:8501
REM ======================================================

REM Use UTF-8 console to avoid Unicode logging errors (checkmarks, bullets)
chcp 65001 >NUL 2>&1
set PYTHONIOENCODING=utf-8

REM Change to project root (this script is inside scripts\)
pushd "%~dp0.."

REM Optional: activate conda env if available
where conda >NUL 2>&1
if %ERRORLEVEL%==0 (
  echo Activating conda environment: superml
  call conda activate superml
) else (
  echo conda not found in PATH. Continuing with default Python.
)

REM Ensure local SQLite DB is used for development
set DATABASE_URL=sqlite:///data/customer_insights.db

REM Create data directory if missing
if not exist data mkdir data

REM Start API (port 8000) - runs DB setup before server
start "CICOP API" cmd /k "echo Starting API... && python scripts\setup_database.py && python scripts\generate_standalone_data.py && echo API: http://localhost:8000 && uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload"

REM Start ML Model Server (port 8001)
start "ML Server" cmd /k "echo Starting ML Model Server... && echo ML: http://localhost:8001 && uvicorn src.ml.model_server:app --host 0.0.0.0 --port 8001 --reload"

REM Start API Gateway (port 8002)
start "API Gateway" cmd /k "echo Starting API Gateway... && echo Gateway: http://localhost:8002 && uvicorn src.api.gateway:app --host 0.0.0.0 --port 8002 --reload"

REM Start Streamlit Dashboard (port 8501)
REM Ensure Python can import the src package when Streamlit launches
set PYTHONPATH=%CD%
start "Dashboard" cmd /k "echo Starting Streamlit Dashboard... && echo Dashboard: http://localhost:8501 && streamlit run src\dashboard\app.py --server.port=8501 --server.address=0.0.0.0"

popd
endlocal
exit /b 0
