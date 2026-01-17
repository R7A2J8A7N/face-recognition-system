# scripts/enroll_users.py

import os
import cv2
from pathlib import Path

from src.detector import FaceDetector
from src.quality import FaceQualityChecker
from src.embedder import FaceEmbedder
from src.database import FaceDatabase


# =========================
# CONFIGURATION
# =========================

ENROLL_DIR = Path("data/enroll")
VECTOR_DB_PATH = "vector_db/chroma_faces"

# Minimum images per user recommended (soft rule)
MIN_IMAGES_PER_USER = 3


# =========================
# INITIALIZE COMPONENTS
# =========================

detector = FaceDetector()
quality_checker = FaceQualityChecker()
embedder = FaceEmbedder()
db = FaceDatabase(path=VECTOR_DB_PATH)


# =========================
# HELPER FUNCTIONS
# =========================

def enroll_single_user(user_id: str, user_folder: Path):
    """
    Enroll a single user by processing all images in their folder.

    Edge cases handled:
    - Empty folder
    - Image read failure
    - No face detected
    - Low quality face
    """

    image_files = list(user_folder.glob("*"))

    if len(image_files) < MIN_IMAGES_PER_USER:
        print(f"[WARN] User {user_id} has very few images ({len(image_files)})")

    success_count = 0

    for img_path in image_files:
        image = cv2.imread(str(img_path))

        if image is None:
            print(f"[SKIP] Cannot read image: {img_path}")
            continue

        faces = detector.detect(image)

        # Edge case: no face detected
        if not faces:
            print(f"[SKIP] No face found in {img_path}")
            continue

        # NOTE: Enrollment assumes ONE face per image
        face = faces[0]

        # Quality check
        if not quality_checker.is_valid(image, face):
            print(f"[SKIP] Low quality face in {img_path}")
            continue

        embedding = embedder.get_embedding(face)

        if embedding is None:
            print(f"[SKIP] Failed to get embedding for {img_path}")
            continue

        # Store embedding with metadata
        db.add_embedding(
            embedding=embedding,
            user_id=user_id,
            meta={
                "image_name": img_path.name,
                "source": "enroll"
            }
        )

        success_count += 1

    print(f"[DONE] User {user_id}: {success_count} embeddings added")


# =========================
# MAIN ENROLLMENT LOGIC
# =========================

def enroll_all_users():
    """
    Enroll all users found in the ENROLL_DIR.
    """

    if not ENROLL_DIR.exists():
        raise FileNotFoundError(
            f"Enrollment directory not found: {ENROLL_DIR}"
        )

    user_folders = [
        f for f in ENROLL_DIR.iterdir()
        if f.is_dir()
    ]

    if not user_folders:
        print("[ERROR] No user folders found for enrollment")
        return

    print(f"[INFO] Found {len(user_folders)} users for enrollment")

    for user_folder in user_folders:
        user_id = user_folder.name
        print(f"\n[INFO] Enrolling user: {user_id}")
        enroll_single_user(user_id, user_folder)

    print("\n[SUCCESS] Enrollment completed for all users")


# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":
    enroll_all_users()
