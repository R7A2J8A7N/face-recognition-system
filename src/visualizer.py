# src/visualizer.py
import cv2


class FaceVisualizer:
    """
    Draws bounding boxes and labels on image.
    """

    def draw(self, image, face, label, color):
        x1, y1, x2, y2 = map(int, face.bbox)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        (w, h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(image, (x1, y1 - h - 10), (x1 + w, y1), color, -1)

        cv2.putText(
            image, label, (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (255, 255, 255), 2
        )
