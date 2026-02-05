from insightface.app import FaceAnalysis
from insightface.app.common import Face
import numpy as np
from typing import List
from src.config.settings import settings


class FaceDetector:

    def __init__(self) -> None:

        ctx_id = 0 if "CUDAExecutionProvider" in settings.MODEL_PROVIDERS else -1

        self.app = FaceAnalysis(
            name=settings.FACE_MODEL_NAME,
            providers=settings.MODEL_PROVIDERS,
            allowed_modules=['detection', 'recognition']
        )

        self.app.prepare(
            ctx_id=ctx_id,
            det_size=settings.DET_SIZE
        )

        # Warmup
        if settings.MODEL_WARMUP:
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            try:
                self.app.get(dummy)
            except Exception:
                pass

    def detect(self, image: np.ndarray) -> List[Face]:

        if image is None or image.size == 0:
            return []

        h, w = image.shape[:2]

        if max(h, w) > settings.MAX_IMAGE_DIMENSION:
            raise ValueError("Image too large.")

        faces = self.app.get(image)

        return faces[:settings.MAX_FACES_PER_IMAGE]
