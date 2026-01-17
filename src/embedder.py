# src/embedder.py
import numpy as np


class FaceEmbedder:
    """
    Generates embeddings from detected faces.
    """

    def get_embedding(self, face):
        """
        Extract embedding from a detected face.

        Returns:
            np.ndarray or None
        """
        if face is None or not hasattr(face, "embedding"):
            return None

        return face.embedding
