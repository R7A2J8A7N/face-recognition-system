import chromadb
import numpy as np
import uuid
from typing import List, Dict, Any, Optional
from src.config.settings import settings


class FaceDatabase:

    def __init__(self, path: Optional[str] = None) -> None:

        self.client = chromadb.PersistentClient(
            path=path or settings.DB_PATH
        )

        self.collection = self.client.get_or_create_collection(
            name=settings.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add_embedding(
        self,
        embedding: np.ndarray,
        user_id: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:

        if embedding is None:
            raise ValueError("Embedding is None.")

        if embedding.ndim != 1:
            raise ValueError("Embedding must be 1D.")

        if np.isnan(embedding).any():
            raise ValueError("Embedding contains NaNs.")

        # normalize
        norm = np.linalg.norm(embedding)

        if norm == 0:
            raise ValueError("Zero embedding detected. Rejecting.")

        embedding = embedding / norm


        self.collection.add(
            ids=[str(uuid.uuid4())],
            embeddings=[embedding.astype(np.float32).tolist()],
            metadatas=[{"user_id": user_id, **(meta or {})}],
        )

    def search(
        self,
        embedding: np.ndarray,
        top_k: int = settings.TOP_K,
    ) -> List[Dict[str, Any]]:

        embedding = embedding / np.linalg.norm(embedding)

        result = self.collection.query(
            query_embeddings=[embedding.astype(np.float32).tolist()],
            n_results=top_k,
        )

        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        matches: List[Dict[str, Any]] = []

        for meta, dist in zip(metadatas, distances):

            matches.append({
                "user_id": meta.get("user_id"),
                "distance": float(dist),
                "meta": meta,
            })
        print(result["distances"])

        return matches
    
    def list_all_embeddings(self) -> List[Dict[str, Any]]:
        result = self.collection.get(
            include=["metadatas", "embeddings"]
        )

        metadatas = result.get("metadatas", [])
        embeddings = result.get("embeddings", [])

        records: List[Dict[str, Any]] = []

        for meta, emb in zip(metadatas, embeddings):

            vector_dim = len(emb) if emb is not None else 0

            records.append({
                "user_id": meta.get("user_id"),
                "image": meta.get("image"),
                "vector_dim": vector_dim
            })

        return records
    def delete_user(self, user_id: str) -> None:
        """
    Deletes all embeddings for a user.
    Used for safe re-enrollment.
    """

        try:
            self.collection.delete(
            where={"user_id": user_id}
        )
        except Exception:
        # Never let deletion crash enrollment
            pass
    
    def user_exists(self, user_id: str) -> bool:
        """
        Fast existence check.

    Avoids loading embeddings into memory.
    O(1) lookup.
    """

        result = self.collection.get(
        where={"user_id": user_id},
        limit=1
    )

        ids = result.get("ids")

        return bool(ids)
