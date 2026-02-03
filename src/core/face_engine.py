from typing import List, Dict, Any
import cv2
from pathlib import Path

from src.core.detector import FaceDetector
from src.core.quality import FaceQualityChecker
from src.core.embedder import FaceEmbedder
from src.db.database import FaceDatabase
from src.core.matcher import FaceMatcher
from src.core.confidence import distance_to_confidence


class FaceEngine:

    def __init__(self, db_path: str = "./vector_db") -> None:

        self.detector = FaceDetector()
        self.quality = FaceQualityChecker()
        self.embedder = FaceEmbedder()
        self.db = FaceDatabase(db_path)
        self.matcher = FaceMatcher()

    # ---------------------------
    # Batch Enrollment
    # ---------------------------

    def enroll_dataset(self, dataset_path: str) -> int:

        dataset = Path(dataset_path)

        total = 0

        for user_folder in dataset.iterdir():

            if not user_folder.is_dir():
                continue

            user_id = user_folder.name

            for img_path in user_folder.glob("*.*"):

                image = cv2.imread(str(img_path))

                if image is None:
                    continue

                faces = self.detector.detect(image)

                if not faces:
                    continue

                face = faces[0]

                if not self.quality.is_valid(image, face):
                    continue

                emb = self.embedder.get_embedding(face)

                if emb is None:
                    continue

                self.db.add_embedding(
                    emb,
                    user_id,
                    meta={"image": img_path.name}
                )

                total += 1

        return total

    # ---------------------------
    # Recognition
    # ---------------------------

    def recognize(self, image) -> List[Dict[str, Any]]:

        faces = self.detector.detect(image)

        outputs = []

        for face in faces:

            if not self.quality.is_valid(image, face):
                continue

            emb = self.embedder.get_embedding(face)

            if emb is None:
                continue

            matches = self.db.search(emb)

            user, dist, decision = self.matcher.match(matches)

            outputs.append({
                "user_id": user,
                "confidence": distance_to_confidence(dist),
                "distance": dist,
                "decision": decision,
                "bbox": face.bbox.tolist(),
                "matched_image":
                    matches[0]["meta"]["image"] if matches else None
            })

        return outputs
