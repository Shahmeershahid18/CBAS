"""Pydantic request schema for the CRM-native lead-scoring endpoint."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class CrmLeadFeatures(BaseModel):
    """Features engineered by the app from a CRM lead record. All optional — the model imputes the rest."""
    source: Optional[str] = Field(None, description="Lead source (e.g. Referral, Website, Cold Call)")
    service: Optional[str] = Field(None, description="Service the lead is interested in")
    quotation: Optional[float] = Field(None, description="Quoted deal value")
    num_activities: Optional[float] = Field(None, description="Number of logged activities on the lead")
    has_email: Optional[float] = Field(None, description="1 if an email is on file else 0")
    has_phone: Optional[float] = Field(None, description="1 if a phone is on file else 0")

    model_config = {"extra": "ignore"}
