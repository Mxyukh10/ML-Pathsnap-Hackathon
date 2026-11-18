
from pydantic import BaseModel, Field
from typing import Any


class PredictionRequest(BaseModel):
    trip_id: str = Field(..., min_length=1)


class PredictionResponse(BaseModel):
    trip_id: str
    status: str
    route_coordinates: list[list[float]]
    predicted_arc_ids: list[int]
    metrics: dict[str, Any]
    inference_time_seconds: float
