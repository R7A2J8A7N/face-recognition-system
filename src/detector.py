# src/detector.py
import cv2
from insightface.app import FaceAnalysis


class FaceDetector:
    """
    Handles face detection using InsightFace.
    This class should ONLY detect faces, nothing else.
    """

    def __init__(self, det_size=(640, 640)):
        self.app = FaceAnalysis(providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=det_size)

    def detect(self, image):
        """
        Detect faces in an image.

        Args:
            image (np.ndarray): BGR image from OpenCV

        Returns:
            list: list of detected face objects
        """
        if image is None:
            # Edge case: invalid image path
            return []

        faces = self.app.get(image)

        # Edge case: no faces
        if not faces:
            return []

        return faces
