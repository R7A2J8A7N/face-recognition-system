from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path

from src.config.settings import settings
from src.utils.image_loader import load_image

from src.core.detector import FaceDetector
from src.core.quality import FaceQualityChecker
from src.core.embedder import FaceEmbedder
from src.db.database import FaceDatabase
from src.core.matcher import FaceMatcher
from src.core.confidence import distance_to_confidence


class FaceEngine:
    """
    Core inference engine.

    Responsibilities:
    - Detection
    - Quality filtering
    - Embedding
    - Vector search
    - Matching

    Designed to be loaded ONCE per process.
    """

    def __init__(self) -> None:

        self.detector = FaceDetector()
        self.quality = FaceQualityChecker()
        self.embedder = FaceEmbedder()
        self.db = FaceDatabase(settings.DB_PATH)
        self.matcher = FaceMatcher()

        if settings.MODEL_WARMUP:
            self._warmup()

    # -------------------------------------------------
    # Warmup
    # -------------------------------------------------

    def _warmup(self) -> None:

        dummy = np.zeros((640, 640, 3), dtype=np.uint8)

        try:
            self.detector.detect(dummy)
        except Exception:
            pass

    # -------------------------------------------------
    # Batch Enrollment (PRODUCTION VERSION)
    # -------------------------------------------------

    def enroll_dataset(self, dataset_path: str) -> Dict[str, Any]:

        dataset = Path(dataset_path)

        if not dataset.exists():
            raise ValueError(f"Dataset not found: {dataset}")

        report: Dict[str, Any] = {}

        for user_folder in dataset.iterdir():

            if not user_folder.is_dir():
                continue

            user_id = user_folder.name

            # 🔥 CRITICAL → prevent duplicate vectors
            self.db.delete_user(user_id)

            stored = 0
            skipped_no_face = 0
            skipped_quality = 0
            skipped_embedding = 0

            for img_path in user_folder.glob("*.*"):

                image = load_image(str(img_path))

                faces = self.detector.detect(image)

                if not faces:
                    skipped_no_face += 1
                    continue

                # workload protection
                faces = faces[:settings.MAX_FACES_PER_IMAGE]

                # ⭐ pick largest face instead of rejecting multi-face images
                face = max(
                    faces,
                    key=lambda f:
                    (f.bbox[2] - f.bbox[0]) *
                    (f.bbox[3] - f.bbox[1])
                )

                if not self.quality.is_valid(image, face):
                    skipped_quality += 1
                    continue

                emb = self.embedder.get_embedding(face)

                if emb is None:
                    skipped_embedding += 1
                    continue

                self.db.add_embedding(
                    emb,
                    user_id,
                    meta={"image": img_path.name}
                )

                stored += 1

            # 🚨 Identity Stability Check
            if stored < settings.MIN_EMBEDDINGS_PER_USER:

                # rollback weak identity
                self.db.delete_user(user_id)

                report[user_id] = {
                    "status": "FAILED",
                    "reason": "insufficient_embeddings",
                    "stored": stored,
                    "required": settings.MIN_EMBEDDINGS_PER_USER,
                    "skipped_no_face": skipped_no_face,
                    "skipped_quality": skipped_quality,
                    "skipped_embedding": skipped_embedding,
                }

            else:

                report[user_id] = {
                    "status": "ENROLLED",
                    "stored": stored,
                    "skipped_no_face": skipped_no_face,
                    "skipped_quality": skipped_quality,
                    "skipped_embedding": skipped_embedding,
                }

        return report

    # -------------------------------------------------
    # Recognition
    # -------------------------------------------------

    def recognize(self, image: np.ndarray) -> List[Dict[str, Any]]:

        faces = self.detector.detect(image)

        if not faces:
            return []

        faces = faces[:settings.MAX_FACES_PER_IMAGE]

        outputs: List[Dict[str, Any]] = []

        for face in faces:

            if not self.quality.is_valid(image, face):
                continue

            emb = self.embedder.get_embedding(face)

            if emb is None:
                continue

            matches = self.db.search(emb)

            user, dist, decision = self.matcher.match(matches)

            matched_image: Optional[str] = None

            if matches:
                meta = matches[0].get("meta")
                if meta:
                    matched_image = meta.get("image")

            outputs.append({
                "user_id": user,
                "confidence": distance_to_confidence(dist),
                "distance": dist,
                "decision": decision,
                "bbox": face.bbox.tolist(),
                "matched_image": matched_image
            })

        return outputs

    # -------------------------------------------------
    # Convenience Helper
    # -------------------------------------------------

    def recognize_from_path(self, path: str) -> List[Dict[str, Any]]:
        image = load_image(path)
        return self.recognize(image)

    # -------------------------------------------------
    # Admin
    # -------------------------------------------------

    def list_embeddings(self) -> List[Dict[str, Any]]:
        return self.db.list_all_embeddings()
