from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    DB_PATH: str = "./vector_db"

    DET_SIZE: tuple[int, int] = (640, 640)

    MATCH_THRESHOLD: float = 0.35
    UNCERTAIN_THRESHOLD: float = 0.45

    BLUR_THRESHOLD: float = 80.0
    MIN_FACE_SIZE: int = 50


settings = Settings()
