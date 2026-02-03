import cv2
import numpy as np
from typing import List, Dict, Any


def draw_results(
    image: np.ndarray,
    results: List[Dict[str, Any]]
) -> np.ndarray:
    """
    Draw bounding boxes + labels on image.
    Returns modified image.
    """

    img = image.copy()

    for r in results:

        bbox = r.get("bbox")
        user = r.get("user_id") or "Unknown"
        confidence = r.get("confidence", 0.0)
        decision = r.get("decision", "UNKNOWN")

        if not bbox:
            continue

        x1, y1, x2, y2 = map(int, bbox)

        # Choose color
        if decision == "MATCH":
            color = (0, 255, 0)      # green
        elif decision == "UNCERTAIN":
            color = (0, 255, 255)    # yellow
        else:
            color = (0, 0, 255)      # red

        label = f"{user} ({confidence:.2f})"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        (w, h), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            2
        )

        cv2.rectangle(
            img,
            (x1, y1 - h - 10),
            (x1 + w, y1),
            color,
            -1
        )

        cv2.putText(
            img,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    return img
