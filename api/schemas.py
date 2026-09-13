from pydantic import BaseModel
from typing import List, Optional

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

class Detection(BaseModel):
    class_name: str
    confidence: float
    box: BoundingBox

class DetectionResponse(BaseModel):
    detections: List[Detection]
    image_width: int
    image_height: int

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
