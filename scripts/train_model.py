# scripts/train_model.py
"""
CLI to train and persist a production model artifact used by the ML Model Server.
"""
import sys
import os

# Ensure src is importable when run from repo root
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.ml.training import main as train_main  # noqa: E402

if __name__ == "__main__":
    train_main()
