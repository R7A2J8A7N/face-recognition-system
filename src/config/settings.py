from pydantic_settings import BaseSettings
from pydantic import Field, model_validator
from typing import List


class Settings(BaseSettings):

    # -----------------------------
    # Database
    # -----------------------------
    DB_PATH: str = "./vector_db"
    COLLECTION_NAME: str = "faces"

    # -----------------------------
    # Model
    # -----------------------------
    FACE_MODEL_NAME: str = "buffalo_l"
    USE_GPU: bool = False

    # -----------------------------
    # Detection
    # -----------------------------
    DET_SIZE: tuple[int, int] = (640, 640)
    MIN_FACE_SIZE: int = 50
    MIN_FACE_AREA: int = 2500
    MIN_DET_SCORE: float = 0.6
    BLUR_THRESHOLD: float = 80.0
    # -----------------------------
# Model Runtime Behavior
# -----------------------------
    MODEL_WARMUP: bool = True

    # -----------------------------
    # Matcher
    # -----------------------------
    MIN_EMBEDDINGS_PER_USER: int = 1
    MATCH_THRESHOLD: float = 0.35
    UNCERTAIN_THRESHOLD: float = 0.45
    MIN_VOTES: int = 1
    TOP_K: int = 5
    DISTANCE_EPSILON: float = 1e-6

    # -----------------------------
# Runtime Safety Limits
# -----------------------------
    MODEL_WARMUP: bool = True

    MAX_IMAGE_DIMENSION: int = 4096
    MAX_FACES_PER_IMAGE: int = 5
    MAX_FACE_ANGLE: float = 35.0

    EMBEDDING_DIM: int = 512

    HARD_REJECT_THRESHOLD: float = 0.65
    MIN_SIMILARITY_MARGIN: float = 0.05

    # -----------------------------
    # API Safety (future-proof)
    # -----------------------------
    MAX_IMAGE_SIZE_MB: int = 5
    LOG_LEVEL: str = "INFO"

    # -----------------------------
    # Providers
    # -----------------------------
    MODEL_PROVIDERS: List[str] = Field(
        default_factory=lambda: ["CPUExecutionProvider"]
    )

    # -----------------------------
    # Validators
    # -----------------------------
    @model_validator(mode="after")
    def validate_thresholds(self):

        if self.MATCH_THRESHOLD >= self.UNCERTAIN_THRESHOLD:
            raise ValueError(
                "MATCH_THRESHOLD must be lower than UNCERTAIN_THRESHOLD"
            )

        return self

    # -----------------------------
    # Pydantic Behavior
    # -----------------------------
    model_config = {
        "extra": "forbid",
        "frozen": True
    }


settings = Settings()
