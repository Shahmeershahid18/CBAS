"""Pydantic request/response schemas for the churn endpoint."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ChurnFeatures(BaseModel):
    """All fields optional — the CRM sends what it knows, the model imputes the rest."""
    tenure: Optional[float] = Field(None, description="Months as a customer")
    monthly_charges: Optional[float] = Field(None, description="Monthly charge amount")
    total_charges: Optional[float] = Field(None, description="Lifetime total charges")
    contract: Optional[str] = Field(None, description="Month-to-month / One year / Two year")
    payment_method: Optional[str] = Field(None, description="Payment method")
    internet_service: Optional[str] = Field(None, description="DSL / Fiber optic / No")
    paperless_billing: Optional[str] = Field(None, description="Yes / No")
    tech_support: Optional[str] = Field(None, description="Yes / No")
    online_security: Optional[str] = Field(None, description="Yes / No")
    senior_citizen: Optional[str] = Field(None, description="'0' or '1'")

    model_config = {"extra": "ignore"}


class KeyFactor(BaseModel):
    feature: str
    importance: float


class ChurnResponse(BaseModel):
    churn_probability: float = Field(..., ge=0, le=1)
    risk_score: int = Field(..., ge=0, le=100)
    risk_band: str = Field(..., description="Low / Medium / High")
    model: str
    reason: str
    key_factors: list[KeyFactor]
