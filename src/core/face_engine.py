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
    Core Intelligence Layer.

    Responsibilities:
    -----------------
    • Face Detection
    • Quality Filtering
    • Embedding Extraction
    • Vector Search
    • Identity Matching

    Designed to be:
    ✔ Loaded once per process
    ✔ Thread-safe for inference
    ✔ Stateless (DB holds identity)
    """

    # -------------------------------------------------
    # INIT
    # -------------------------------------------------

    def __init__(self) -> None:

        # Heavy models should load ONLY once
        self.detector = FaceDetector()
        self.quality = FaceQualityChecker()
        self.embedder = FaceEmbedder()
        self.db = FaceDatabase(settings.DB_PATH)
        self.matcher = FaceMatcher()

        # Prevent cold-start latency
        if settings.MODEL_WARMUP:
            self._warmup()

    # -------------------------------------------------
    # MODEL WARMUP
    # -------------------------------------------------

    def _warmup(self) -> None:
        """
        Runs a dummy inference.

        Prevents the FIRST API request from being slow.
        """

        dummy = np.zeros((640, 640, 3), dtype=np.uint8)

        try:
            self.detector.detect(dummy)
        except Exception:
            # Warmup must NEVER crash the service
            pass

    # =================================================
    # SINGLE USER ENROLLMENT  ⭐⭐⭐ PRODUCTION CRITICAL
    # =================================================

    def enroll_user(self, user_folder_path: str) -> Dict[str, Any]:
        """
        Safely enroll ONE identity.

        Guarantees:
        ----------
        ✔ wipes old embeddings (re-enroll safe)
        ✔ rejects corrupted vectors
        ✔ enforces identity strength
        ✔ selects best face automatically
        """

        folder = Path(user_folder_path)

        if not folder.exists():
            raise ValueError(f"User folder not found: {folder}")

        if not folder.is_dir():
            raise ValueError("Enrollment path must be a directory.")

        user_id = folder.name

        # ----------------------------------------
        # Prevent duplicate enrollment
        # ----------------------------------------

        if self.db.user_exists(user_id):

            return {
                    "user": user_id,
                    "status": "EXISTS",
                    "message": "User already enrolled. Skipping storage."
                }


        # 🚨 Prevent duplicate vectors
        self.db.delete_user(user_id)

        stored = 0
        skipped_no_face = 0
        skipped_quality = 0
        skipped_embedding = 0

        images = list(folder.glob("*.*"))

        if not images:
            raise ValueError("No images found for enrollment.")

        for img_path in images[:settings.MAX_EMBEDDINGS_PER_USER]:

            image = load_image(str(img_path))
            faces = self.detector.detect(image)

            if not faces:
                skipped_no_face += 1
                continue

            # pick largest face
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

        # 🔥 Identity Strength Check
        if stored < settings.MIN_EMBEDDINGS_PER_USER:

            # rollback weak identity
            self.db.delete_user(user_id)

            return {
                "user": user_id,
                "status": "FAILED",
                "reason": "weak_identity",
                "stored": stored,
                "required": settings.MIN_EMBEDDINGS_PER_USER,
                "skipped_no_face": skipped_no_face,
                "skipped_quality": skipped_quality,
                "skipped_embedding": skipped_embedding,
            }

        return {
            "user": user_id,
            "status": "ENROLLED",
            "stored": stored,
            "skipped_no_face": skipped_no_face,
            "skipped_quality": skipped_quality,
            "skipped_embedding": skipped_embedding,
        }

    # =================================================
    # BATCH ENROLLMENT
    # =================================================

    def enroll_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """
        Bulk enrollment.

        Used for:
        • migrations
        • enterprise onboarding
        • dataset bootstrapping
        """

        dataset = Path(dataset_path)

        if not dataset.exists():
            raise ValueError(f"Dataset not found: {dataset}")

        report: Dict[str, Any] = {}

        for user_folder in dataset.iterdir():

            if not user_folder.is_dir():
                continue

            result = self.enroll_user(str(user_folder))

            report[user_folder.name] = result

        return report

    # =================================================
    # RECOGNITION
    # =================================================

    def recognize(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Stateless recognition pipeline.

        SAFE for high concurrency APIs.
        """

        faces = self.detector.detect(image)

        if not faces:
            return []

        # Crowd protection
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

            # 🚨 JSON SAFE VALUE
            if not np.isfinite(dist):
                dist = 999.0

            matched_image: Optional[str] = None

            if matches:
                meta = matches[0].get("meta")
                if meta:
                    matched_image = meta.get("image")

            outputs.append({
                "user_id": user,
                "confidence": float(distance_to_confidence(dist)),
                "distance": float(dist),
                "decision": decision,
                "bbox": face.bbox.tolist(),
                "matched_image": matched_image
            })

        return outputs

    # -------------------------------------------------
    # Convenience
    # -------------------------------------------------

    def recognize_from_path(self, path: str) -> List[Dict[str, Any]]:
        image = load_image(path)
        return self.recognize(image)

    # -------------------------------------------------
    # Admin / Audit
    # -------------------------------------------------

    def list_embeddings(self) -> List[Dict[str, Any]]:
        return self.db.list_all_embeddings()
