"""Pydantic request/response schemas for the lead-scoring endpoint."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class LeadFeatures(BaseModel):
    """
    All fields optional — the CRM sends whatever it knows and the model imputes
    the rest. Keys mirror app.models.lead_features.ALL_FEATURES.
    """
    total_visits: Optional[float] = Field(None, description="Number of site visits")
    time_on_site: Optional[float] = Field(None, description="Total time on website (seconds)")
    page_views_per_visit: Optional[float] = Field(None, description="Avg page views per visit")
    lead_origin: Optional[str] = Field(None, description="How the lead entered the system")
    lead_source: Optional[str] = Field(None, description="Traffic/referral source")
    do_not_email: Optional[str] = Field(None, description="'Yes' or 'No'")
    last_activity: Optional[str] = Field(None, description="Most recent lead activity")
    occupation: Optional[str] = Field(None, description="Current occupation")
    specialization: Optional[str] = Field(None, description="Area of interest")

    model_config = {"extra": "ignore"}


class KeyFactor(BaseModel):
    feature: str
    importance: float


class ScoreResponse(BaseModel):
    score: int = Field(..., ge=0, le=100, description="0-100 lead score")
    probability: float = Field(..., ge=0, le=1)
    band: str = Field(..., description="Hot / Warm / Cold")
    model: str
    reason: str
    key_factors: list[KeyFactor]
