# ui_enroll.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image

from src.detector import FaceDetector
from src.quality import FaceQualityChecker
from src.embedder import FaceEmbedder
from src.database import FaceDatabase

# -----------------------
# CONFIG
# -----------------------

IMAGE_TYPES = [
    ("Front", "front", True),
    ("Left", "left", True),
    ("Right", "right", True),
    ("Smile", "smile", False),
    ("Glasses", "glasses", False),
    ("Low Light", "low_light", False),
    ("Bright Light", "bright_light", False),
    ("Slight Blur", "slight_blur", False),
]

# -----------------------
# INIT
# -----------------------

detector = FaceDetector()
quality = FaceQualityChecker()
embedder = FaceEmbedder()
db = FaceDatabase(path="vector_db/chroma_faces")

st.set_page_config(page_title="Face Enrollment", layout="wide")

# -----------------------
# HEADER
# -----------------------

st.title("🧑 Face Enrollment UI (1K Users Ready)")
st.caption("Upload multiple face variations for better recognition accuracy")

st.markdown("### ✅ Mandatory (Minimum Required)")
st.success("Front • Left • Right")

user_id = st.text_input("Enter User ID (e.g. user_0001)")

st.divider()

# -----------------------
# GRID UI
# -----------------------

rows = [IMAGE_TYPES[:4], IMAGE_TYPES[4:]]

for row in rows:
    cols = st.columns(4)
    for col, (label, key, is_mandatory) in zip(cols, row):
        with col:
            st.subheader(label)

            if is_mandatory:
                st.markdown("🟥 **Mandatory**")
            else:
                st.markdown("🟦 Optional")

            uploaded_file = st.file_uploader(
                f"Upload {label}",
                type=["jpg", "jpeg", "png"],
                key=f"{key}_uploader"
            )

            if uploaded_file and user_id:
                image = Image.open(uploaded_file).convert("RGB")
                img_np = np.array(image)
                img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                st.image(image, width=True)

                if st.button(f"Enroll {label}", key=f"{key}_btn"):
                    faces = detector.detect(img_cv)

                    if not faces:
                        st.error("No face detected")
                        continue

                    face = faces[0]

                    if not quality.is_valid(img_cv, face):
                        st.error("Low quality face")
                        continue

                    embedding = embedder.get_embedding(face)
                    if embedding is None:
                        st.error("Embedding failed")
                        continue

                    db.add_embedding(
                        embedding=embedding,
                        user_id=user_id,
                        meta={
                            "image_type": key,
                            "mandatory": is_mandatory,
                            "source": "ui"
                        }
                    )

                    st.success(f"{label} image enrolled")

st.divider()
st.caption("⚠️ Minimum required images: Front, Left, Right")
