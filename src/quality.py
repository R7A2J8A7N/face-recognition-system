# src/quality.py
import cv2
import numpy as np


class FaceQualityChecker:
    """
    Filters out low-quality face detections.
    """

    def __init__(self, min_face_size=50, min_score=0.5, blur_threshold=80):
        self.min_face_size = min_face_size
        self.min_score = min_score
        self.blur_threshold = blur_threshold

    def is_blurry(self, face_img):
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance < self.blur_threshold

    def is_valid(self, image, face):
        x1, y1, x2, y2 = map(int, face.bbox)
        w, h = x2 - x1, y2 - y1

        # Edge case: very small face
        if w < self.min_face_size or h < self.min_face_size:
            return False

        # Edge case: low detection confidence
        if hasattr(face, "det_score") and face.det_score < self.min_score:
            return False

        face_img = image[y1:y2, x1:x2]

        # Edge case: blur
        if self.is_blurry(face_img):
            return False

        return True
