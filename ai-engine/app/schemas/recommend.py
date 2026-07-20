"""Pydantic response schema for the recommendations endpoint."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class RecommendedItem(BaseModel):
    item_id: int
    name: str
    category: Optional[str] = None
    score: Optional[float] = Field(None, description="Predicted affinity; null for popularity fallback")


class RecommendationsResponse(BaseModel):
    customer_id: str
    personalized: bool = Field(..., description="False = cold-start popularity fallback")
    recommendations: list[RecommendedItem]
