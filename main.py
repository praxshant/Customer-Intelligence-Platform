# main.py
# Purpose: Entry point for the Streamlit application

import streamlit as st
import sys
import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.dashboard.app import main as dashboard_main
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

"""
Expose a FastAPI application for production deployments.
Dockerfile (production stage) runs: gunicorn main:app ...
This FastAPI app provides at least a /health endpoint for health checks.
"""
app = FastAPI(title="CICOP Main App", version="0.1.0")

@app.get("/health")
async def health():
    return JSONResponse({"status": "ok", "service": "main"})

def main():
    # Step 1: Configure Streamlit page
    st.set_page_config(
        page_title="Customer Insights Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Step 2: Load configuration
    try:
        config = load_config()
    except Exception as e:
        st.error(f"Configuration error: {str(e)}")
        return
    
    # Step 3: Setup logging
    logger = setup_logger("main")
    logger.info("Starting Customer Insights Dashboard")
    
    # Step 4: Run the main dashboard application
    try:
        dashboard_main()
    except Exception as e:
        logger.error(f"Error running dashboard: {str(e)}")
        st.error("An error occurred while loading the dashboard. Please check the logs.")

if __name__ == "__main__":
    main()