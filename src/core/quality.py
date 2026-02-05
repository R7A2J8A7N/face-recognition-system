import cv2
import numpy as np
from insightface.app.common import Face
from src.config.settings import settings


class FaceQualityChecker:

    def is_blurry(self, face_img: np.ndarray) -> bool:

        if face_img.size == 0:
            return True

        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_32F).var()

        return variance < settings.BLUR_THRESHOLD

    def is_bad_lighting(self, face_img: np.ndarray) -> bool:

        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        mean = gray.mean()

        return mean < 40 or mean > 220

    def is_valid(self, image: np.ndarray, face: Face) -> bool:

        h, w = image.shape[:2]

        x1, y1, x2, y2 = map(int, face.bbox)

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return False

        # detection confidence
        if hasattr(face, "det_score") and face.det_score < settings.MIN_DET_SCORE:
            return False

        # pose filtering
        if hasattr(face, "pose") and face.pose is not None:
            yaw, pitch, roll = face.pose
            if max(abs(yaw), abs(pitch), abs(roll)) > settings.MAX_FACE_ANGLE:
                return False

        # size check
        if min(x2 - x1, y2 - y1) < settings.MIN_FACE_SIZE:
            return False

        # area check
        face_area = (x2 - x1) * (y2 - y1)
        if face_area < settings.MIN_FACE_AREA:
            return False

        face_img = image[y1:y2, x1:x2]

        if self.is_blurry(face_img):
            return False

        if self.is_bad_lighting(face_img):
            return False

        return True
