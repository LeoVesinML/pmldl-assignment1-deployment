"""Pydantic request/response models for the wine-quality API."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

WineType = Literal["red", "white"]


class WineFeatures(BaseModel):
    """Physicochemical measurements of a single wine sample."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "fixed_acidity": 7.0,
                "volatile_acidity": 0.27,
                "citric_acid": 0.36,
                "residual_sugar": 20.7,
                "chlorides": 0.045,
                "free_sulfur_dioxide": 45.0,
                "total_sulfur_dioxide": 170.0,
                "density": 1.001,
                "ph": 3.0,
                "sulphates": 0.45,
                "alcohol": 8.8,
                "wine_type": "white",
            }
        }
    )

    fixed_acidity: float = Field(..., ge=0, le=20, description="Tartaric acid (g/dm^3)")
    volatile_acidity: float = Field(..., ge=0, le=3, description="Acetic acid (g/dm^3)")
    citric_acid: float = Field(..., ge=0, le=2, description="Citric acid (g/dm^3)")
    residual_sugar: float = Field(..., ge=0, le=80, description="Residual sugar (g/dm^3)")
    chlorides: float = Field(..., ge=0, le=1, description="Sodium chloride (g/dm^3)")
    free_sulfur_dioxide: float = Field(..., ge=0, le=300, description="Free SO2 (mg/dm^3)")
    total_sulfur_dioxide: float = Field(..., ge=0, le=500, description="Total SO2 (mg/dm^3)")
    density: float = Field(..., gt=0.9, lt=1.1, description="Density (g/cm^3)")
    ph: float = Field(..., ge=2, le=5, description="pH value")
    sulphates: float = Field(..., ge=0, le=3, description="Potassium sulphate (g/dm^3)")
    alcohol: float = Field(..., ge=5, le=20, description="Alcohol by volume (%)")
    wine_type: WineType = Field("red", description="Colour of the wine")


class BatchRequest(BaseModel):
    """A batch of wine samples (max 1000 items per request)."""

    items: List[WineFeatures] = Field(..., min_length=1, max_length=1000)


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="1 = good quality, 0 = standard quality")
    label: str = Field(..., description="Human readable prediction")
    probability_good: float = Field(..., ge=0, le=1, description="P(good quality)")
    confidence: float = Field(..., ge=0, le=1, description="Probability of the predicted class")
    threshold: float = Field(..., description="Decision threshold applied to the probability")
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    inference_ms: float = Field(..., description="Server-side inference time in milliseconds")


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    count: int
    inference_ms: float


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded"]
    model_loaded: bool
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    trained_at: Optional[str] = None
    api_version: str
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    trained_at: Optional[str] = None
    git_revision: Optional[str] = None
    target: Optional[str] = None
    positive_class_meaning: Optional[str] = None
    metrics: Dict[str, float] = Field(default_factory=dict)
    leaderboard: Dict[str, Any] = Field(default_factory=dict)
    confusion_matrix: List[List[int]] = Field(default_factory=list)
    train_rows: Optional[int] = None
    test_rows: Optional[int] = None
    sklearn_version: Optional[str] = None
    runtime_sklearn_version: Optional[str] = None


class FeatureSchemaResponse(BaseModel):
    numeric_features: List[str]
    categorical_features: List[str]
    wine_types: List[str]
    feature_ranges: Dict[str, Dict[str, float]]
