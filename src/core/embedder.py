import numpy as np


class FaceEmbedder:

    def get_embedding(self, face) -> np.ndarray | None:

        if face is None or not hasattr(face, "embedding"):
            return None

        emb = np.asarray(face.embedding, dtype=np.float32)

        norm = np.linalg.norm(emb)

        if norm == 0:
            return None

        return emb / norm
