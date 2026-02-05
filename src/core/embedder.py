import numpy as np


class FaceEmbedder:

    def get_embedding(self, face) -> np.ndarray | None:
        """
        Returns a normalized float32 embedding.

        Guarantees:
        - unit norm
        - no NaNs
        - no infinite values
        """

        if face is None or not hasattr(face, "embedding"):
            return None

        emb = face.embedding

        if emb is None:
            return None

        # force float32 early
        emb = np.asarray(emb, dtype=np.float32)

        # reject NaN / Inf
        if not np.isfinite(emb).all():
            return None

        norm = np.linalg.norm(emb)

        # reject zero vectors
        if norm < 1e-6:
            return None

        # ✅ normalize
        emb = emb / norm

        return emb
