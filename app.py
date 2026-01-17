# app.py
import cv2
from src.detector import FaceDetector
from src.quality import FaceQualityChecker
from src.embedder import FaceEmbedder
from src.database import FaceDatabase
from src.matcher import FaceMatcher
from src.visualizer import FaceVisualizer
from src.utils import distance_to_confidence

detector = FaceDetector()
quality = FaceQualityChecker()
embedder = FaceEmbedder()
db = FaceDatabase()
matcher = FaceMatcher()
viz = FaceVisualizer()


def recognize(image_path):
    image = cv2.imread(image_path)
    faces = detector.detect(image)

    for face in faces:
        if not quality.is_valid(image, face):
            continue

        emb = embedder.get_embedding(face)
        results = db.search(emb)

        user, score, decision = matcher.match(results)

        if decision == "MATCH":
            label = f"{user} ({(1-score)*100:.1f}%)"
            color = (0, 255, 0)

        elif decision == "UNCERTAIN":
            label = f"Uncertain ({(1-score)*100:.1f}%)"
            color = (0, 255, 255)

        else:
            label = "Unknown"
            color = (0, 0, 255)


        viz.draw(image, face, label, color)

    return image
