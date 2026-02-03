import cv2
import numpy as np
from src.config.settings import settings


class FaceQualityChecker:

    def is_blurry(self, face_img: np.ndarray) -> bool:

        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()

        return variance < settings.BLUR_THRESHOLD

    def is_valid(self, image: np.ndarray, face) -> bool:

        h, w = image.shape[:2]

        x1, y1, x2, y2 = map(int, face.bbox)

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return False

        if (x2 - x1) < settings.MIN_FACE_SIZE:
            return False

        face_img = image[y1:y2, x1:x2]

        if self.is_blurry(face_img):
            return False

        return True
