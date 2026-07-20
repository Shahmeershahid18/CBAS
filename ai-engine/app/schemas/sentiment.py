"""Pydantic request/response schemas for the sentiment endpoint."""
from __future__ import annotations

from pydantic import BaseModel, Field


class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Review / email / feedback text to analyse")


class SentimentResponse(BaseModel):
    sentiment: str = Field(..., description="positive / negative / neutral")
    confidence: float = Field(..., ge=0, le=1)
    scores: dict[str, float] = Field(..., description="Per-class probability")
