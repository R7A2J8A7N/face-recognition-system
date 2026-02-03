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
            name="faces",
            metadata={"hnsw:space": "cosine"},
        )

    def add_embedding(
        self,
        embedding: np.ndarray,
        user_id: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:

        self.collection.add(
            ids=[str(uuid.uuid4())],
            embeddings=[embedding.astype(np.float32)],
            metadatas=[{"user_id": user_id, **(meta or {})}],
        )

    def search(
        self,
        embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:

        result = self.collection.query(
            query_embeddings=[embedding.astype(np.float32)],
            n_results=top_k,
        )

        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        matches: List[Dict[str, Any]] = []

        for meta, dist in zip(metadatas, distances):

            matches.append({
                "user_id": meta.get("user_id"),
                "distance": float(dist),
                "meta": meta,
            })

        return matches
