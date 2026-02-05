from fastapi import UploadFile, HTTPException
import numpy as np
import cv2


def decode_image(file: UploadFile) -> np.ndarray:
    """
    Converts uploaded file -> OpenCV image.

    Critical protections:
    ✔ empty file
    ✔ corrupted image
    ✔ invalid format
    """

    try:
        data = file.file.read()

        if not data:
            raise HTTPException(
                status_code=400,
                detail="Empty file uploaded."
            )

        np_arr = np.frombuffer(data, np.uint8)

        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid image format."
            )

        return image

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Failed to process image."
        )
