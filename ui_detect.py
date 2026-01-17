# ui_detect.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image

from src.detector import FaceDetector

detector = FaceDetector()

st.set_page_config(page_title="Face Detection Test", layout="centered")
st.title("🔍 Face Detection Test")

uploaded_file = st.file_uploader(
    "Upload image to test detection",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    st.image(image, width=350)

    faces = detector.detect(img_cv)

    if not faces:
        st.error("❌ No face detected")
    else:
        st.success(f"✅ Detected {len(faces)} face(s)")
        for i, face in enumerate(faces):
            st.write(f"Face {i+1} confidence: {face.det_score:.2f}")
