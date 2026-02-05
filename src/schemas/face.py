from pydantic import BaseModel
from typing import Optional, List


class RecognitionResult(BaseModel):

    user_id: Optional[str]
    confidence: float
    distance: float
    decision: str
    bbox: List[float]
