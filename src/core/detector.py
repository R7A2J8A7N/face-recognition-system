from insightface.app import FaceAnalysis
import numpy as np
from typing import List
from src.config.settings import settings


class FaceDetector:

    def __init__(self) -> None:

        self.app = FaceAnalysis(
            providers=["CPUExecutionProvider"]
        )

        self.app.prepare(
            ctx_id=-1,
            det_size=settings.DET_SIZE
        )

    def detect(self, image: np.ndarray) -> List:

        if image is None:
            return []

        return self.app.get(image)
