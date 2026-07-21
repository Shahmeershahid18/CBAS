"""
CBAS AI Engine — FastAPI service.

The "AI Engine" box in the CBAS architecture: a standalone Python service that
serves the trained models over REST. The Next.js backend calls these endpoints
and stores the results against CRM records.

Currently serving: Lead Scoring (POST /score-lead), Churn (POST /predict-churn),
Sentiment (POST /analyze-sentiment), Recommendations (GET /recommendations/{id}).

Run:  uvicorn app.main:app --host 127.0.0.1 --port 8088
"""
from __future__ import annotations

from fastapi import Depends, FastAPI, Header, HTTPException, Query, status

from app import config
from app.models import churn, churn_crm, lead_crm, lead_scoring, recommend, sentiment, sentiment_crm
from app.schemas.churn import ChurnFeatures, ChurnResponse
from app.schemas.churn_crm import CrmChurnFeatures
from app.schemas.lead import LeadFeatures, ScoreResponse
from app.schemas.lead_crm import CrmLeadFeatures
from app.schemas.recommend import RecommendationsResponse
from app.schemas.sentiment import SentimentRequest, SentimentResponse

app = FastAPI(
    title="CBAS AI Engine",
    version="0.1.0",
    description="Trained ML models served over REST for the CBAS platform.",
)


def require_api_key(x_ai_engine_key: str | None = Header(default=None)) -> None:
    """Reject calls without the shared secret, unless no key is configured (dev)."""
    if config.API_KEY and x_ai_engine_key != config.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing x-ai-engine-key header.",
        )


@app.get("/health")
def health() -> dict:
    """Liveness + which models are loaded and ready."""
    return {
        "status": "ok",
        "models": {
            "lead_scoring": {
                "ready": lead_scoring.is_ready(),
            },
            "lead_crm": {
                "ready": lead_crm.is_ready(),
            },
            "churn": {
                "ready": churn.is_ready(),
            },
            "churn_crm": {
                "ready": churn_crm.is_ready(),
            },
            "sentiment": {
                "ready": sentiment.is_ready(),
            },
            "sentiment_crm": {
                "ready": sentiment_crm.is_ready(),
            },
            "recommendations": {
                "ready": recommend.is_ready(),
            },
        },
    }


@app.get("/models/lead-scoring", dependencies=[Depends(require_api_key)])
def lead_scoring_info() -> dict:
    """Model metadata + held-out evaluation metrics (for the report/dashboard)."""
    try:
        return lead_scoring.model_info()
    except lead_scoring.ModelNotTrainedError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc))


@app.post("/score-lead", response_model=ScoreResponse,
          dependencies=[Depends(require_api_key)])
def score_lead(features: LeadFeatures) -> ScoreResponse:
    """Score a single lead's conversion likelihood (0-100)."""
    try:
        result = lead_scoring.score_lead(features.model_dump(exclude_none=True))
    except lead_scoring.ModelNotTrainedError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc))
    return ScoreResponse(**result)


@app.get("/models/lead-scoring-crm", dependencies=[Depends(require_api_key)])
def lead_crm_info() -> dict:
    """CRM-native lead-scoring model metadata + held-out metrics."""
    try:
        return lead_crm.model_info()
    except lead_crm.ModelNotTrainedError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc))


@app.post("/score-lead-crm", response_model=ScoreResponse,
          dependencies=[Depends(require_api_key)])
def score_lead_crm(features: CrmLeadFeatures) -> ScoreResponse:
    """Score a lead's conversion likelihood using the model trained on CBAS's own leads."""
    try:
        result = lead_crm.score_lead(features.model_dump(exclude_none=True))
    except lead_crm.ModelNotTrainedError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc))
    return ScoreResponse(**result)


@app.get("/models/churn", dependencies=[Depends(require_api_key)])
def churn_info() -> dict:
    """Churn model metadata + held-out evaluation metrics."""
    try:
        return churn.model_info()
    except churn.ModelNotTrainedError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc))


@app.post("/predict-churn", response_model=ChurnResponse,
          dependencies=[Depends(require_api_key)])
def predict_churn(features: ChurnFeatures) -> ChurnResponse:
    """Predict a single customer's churn risk (Low / Medium / High)."""
    try:
        result = churn.predict_churn(features.model_dump(exclude_none=True))
    except churn.ModelNotTrainedError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc))
    return ChurnResponse(**result)


@app.get("/models/churn-crm", dependencies=[Depends(require_api_key)])
def churn_crm_info() -> dict:
    """CRM-native churn model metadata + held-out metrics."""
    try:
        return churn_crm.model_info()
    except churn_crm.ModelNotTrainedError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc))


@app.post("/predict-churn-crm", response_model=ChurnResponse,
          dependencies=[Depends(require_api_key)])
def predict_churn_crm(features: CrmChurnFeatures) -> ChurnResponse:
    """Predict churn from CRM engagement features (model trained on CBAS's own data)."""
    try:
        result = churn_crm.predict_churn(features.model_dump(exclude_none=True))
    except churn_crm.ModelNotTrainedError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc))
    return ChurnResponse(**result)


@app.get("/models/sentiment", dependencies=[Depends(require_api_key)])
def sentiment_info() -> dict:
    """Sentiment model metadata + held-out evaluation metrics."""
    try:
        return sentiment.model_info()
    except sentiment.ModelNotTrainedError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc))


@app.post("/analyze-sentiment", response_model=SentimentResponse,
          dependencies=[Depends(require_api_key)])
def analyze_sentiment(req: SentimentRequest) -> SentimentResponse:
    """Classify text as positive / negative / neutral with a confidence score."""
    try:
        result = sentiment.analyze_sentiment(req.text)
    except sentiment.ModelNotTrainedError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc))
    return SentimentResponse(**result)


@app.get("/models/sentiment-crm", dependencies=[Depends(require_api_key)])
def sentiment_crm_info() -> dict:
    """CRM-note sentiment model metadata + metrics."""
    try:
        return sentiment_crm.model_info()
    except sentiment_crm.ModelNotTrainedError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc))


@app.post("/analyze-sentiment-crm", response_model=SentimentResponse,
          dependencies=[Depends(require_api_key)])
def analyze_sentiment_crm(req: SentimentRequest) -> SentimentResponse:
    """Classify a CRM note (business language) as positive / negative / neutral."""
    try:
        result = sentiment_crm.analyze_sentiment(req.text)
    except sentiment_crm.ModelNotTrainedError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc))
    return SentimentResponse(**result)


@app.get("/models/recommendations", dependencies=[Depends(require_api_key)])
def recommendations_info() -> dict:
    """Recommender metadata + held-out Precision@K / Recall@K."""
    try:
        return recommend.model_info()
    except recommend.ModelNotTrainedError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc))


@app.get("/recommendations/{customer_id}", response_model=RecommendationsResponse,
         dependencies=[Depends(require_api_key)])
def get_recommendations(customer_id: str, n: int = Query(5, ge=1, le=20)) -> RecommendationsResponse:
    """Top-N recommended items for a customer (personalized, or popularity fallback)."""
    try:
        result = recommend.recommend(customer_id, n)
    except recommend.ModelNotTrainedError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc))
    return RecommendationsResponse(**result)
