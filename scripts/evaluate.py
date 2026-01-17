# scripts/evaluate.py

import cv2
from pathlib import Path

from src.detector import FaceDetector
from src.quality import FaceQualityChecker
from src.embedder import FaceEmbedder
from src.database import FaceDatabase
from src.matcher import FaceMatcher


# =========================
# CONFIG
# =========================

TEST_DIR = Path("data/test")
VECTOR_DB_PATH = "vector_db/chroma_faces"
TOP_K = 10


# =========================
# INIT COMPONENTS
# =========================

detector = FaceDetector()
quality = FaceQualityChecker()
embedder = FaceEmbedder()
db = FaceDatabase(path=VECTOR_DB_PATH)
matcher = FaceMatcher()


# =========================
# METRICS
# =========================

total_images = 0
true_positive = 0
false_positive = 0
false_negative = 0
skipped = 0


# =========================
# EVALUATION LOGIC
# =========================

def evaluate_single_image(image_path: Path, true_user: str):
    """
    Evaluate recognition result for a single image.
    """

    global total_images, true_positive, false_positive, false_negative, skipped

    image = cv2.imread(str(image_path))
    total_images += 1

    if image is None:
        skipped += 1
        return

    faces = detector.detect(image)

    # Edge case: no face
    if not faces:
        skipped += 1
        return

    face = faces[0]

    if not quality.is_valid(image, face):
        skipped += 1
        return

    embedding = embedder.get_embedding(face)
    if embedding is None:
        skipped += 1
        return

    results = db.search(embedding, top_k=TOP_K)
    predicted_user, score = matcher.match(results)

    # Decision analysis
    if predicted_user == true_user:
        true_positive += 1
    elif predicted_user is None:
        false_negative += 1
    else:
        false_positive += 1


def evaluate_all():
    """
    Run evaluation on entire test dataset.
    """

    if not TEST_DIR.exists():
        raise FileNotFoundError("Test directory not found")

    user_folders = [f for f in TEST_DIR.iterdir() if f.is_dir()]

    print(f"[INFO] Evaluating {len(user_folders)} users")

    for user_folder in user_folders:
        true_user = user_folder.name
        images = list(user_folder.glob("*"))

        for img_path in images:
            evaluate_single_image(img_path, true_user)

    # =========================
    # REPORT
    # =========================

    evaluated = total_images - skipped
    accuracy = (true_positive / evaluated) * 100 if evaluated else 0

    print("\n========== EVALUATION REPORT ==========")
    print(f"Total images        : {total_images}")
    print(f"Evaluated images    : {evaluated}")
    print(f"Skipped images      : {skipped}")
    print(f"True Positives      : {true_positive}")
    print(f"False Positives     : {false_positive}")
    print(f"False Negatives     : {false_negative}")
    print(f"Accuracy (%)        : {accuracy:.2f}")
    print("======================================\n")


# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":
    evaluate_all()
