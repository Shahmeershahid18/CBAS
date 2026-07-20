"""Pydantic request schema for the CRM-native churn endpoint."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class CrmChurnFeatures(BaseModel):
    """Engagement features computed by the app from a contact's CRM record."""
    tenure_days: Optional[float] = Field(None, description="Days since the contact was created")
    num_orders: Optional[float] = Field(None, description="Number of orders placed")
    total_spend: Optional[float] = Field(None, description="Lifetime spend")
    num_activities: Optional[float] = Field(None, description="Number of logged activities")
    num_negative_notes: Optional[float] = Field(None, description="Count of negative-sentiment notes")
    avg_sentiment: Optional[float] = Field(None, description="Mean note polarity (-1..1)")
    num_categories: Optional[float] = Field(None, description="Distinct product categories bought")
    has_open_deal: Optional[float] = Field(None, description="1 if an open deal exists else 0")

    model_config = {"extra": "ignore"}
