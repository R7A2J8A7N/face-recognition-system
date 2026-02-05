from fastapi import APIRouter, UploadFile, File, HTTPException
import numpy as np
import cv2

from src.api.dependencies import get_engine

router = APIRouter(prefix="/recognize", tags=["Recognition"])


@router.post("/")
async def recognize_face(file: UploadFile = File(...)):

    contents = await file.read()

    np_img = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(400, "Invalid image")

    engine = get_engine()

    results = engine.recognize(image)

    return {"faces": results}
