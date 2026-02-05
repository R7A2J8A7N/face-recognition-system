from functools import lru_cache
from src.core.face_engine import FaceEngine


@lru_cache(maxsize=1)
def get_engine() -> FaceEngine:
    """
    Creates ONE engine per process.
    Prevents model reload per request.
    """
    return FaceEngine()
