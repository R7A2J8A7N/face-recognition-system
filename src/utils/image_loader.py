from pathlib import Path
import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    """
    Loads image safely from disk.
    """

    img_path = Path(path).resolve()

    if not img_path.exists():
        raise FileNotFoundError(
            f"Image not found: {img_path}"
        )

    image = cv2.imread(str(img_path))

    if image is None:
        raise ValueError(
            f"Failed to decode image: {img_path}"
        )

    return image
