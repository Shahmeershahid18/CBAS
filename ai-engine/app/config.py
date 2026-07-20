"""Runtime configuration for the CBAS AI Engine, loaded from environment/.env."""
import os
from pathlib import Path

from dotenv import load_dotenv

# ai-engine/ project root (this file is ai-engine/app/config.py)
BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

HOST = os.getenv("AI_ENGINE_HOST", "127.0.0.1")
PORT = int(os.getenv("AI_ENGINE_PORT", "8088"))

# Shared secret expected in the `x-ai-engine-key` header. Empty => check disabled.
API_KEY = os.getenv("AI_ENGINE_API_KEY", "").strip()

ARTIFACTS_DIR = BASE_DIR / os.getenv("ARTIFACTS_DIR", "artifacts")

LEAD_SCORING_ARTIFACT = ARTIFACTS_DIR / "lead_scoring.joblib"
CHURN_ARTIFACT = ARTIFACTS_DIR / "churn.joblib"
CHURN_CRM_ARTIFACT = ARTIFACTS_DIR / "churn_crm.joblib"
SENTIMENT_ARTIFACT = ARTIFACTS_DIR / "sentiment.joblib"
RECOMMEND_ARTIFACT = ARTIFACTS_DIR / "recommend.joblib"
