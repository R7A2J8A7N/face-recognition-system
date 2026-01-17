# src/database.py
import chromadb
import uuid


class FaceDatabase:
    """
    Handles vector database operations.
    """

    def __init__(self, path=None):
        if path:
            self.client = chromadb.PersistentClient(path=path)
        else:
            self.client = chromadb.Client()

        self.collection = self.client.get_or_create_collection(
            name="faces",
            metadata={"hnsw:space": "cosine"}
        )

    def add_embedding(self, embedding, user_id, meta=None):
        """
        Store an embedding for a user.
        """
        self.collection.add(
            ids=[str(uuid.uuid4())],
            embeddings=[embedding.tolist()],
            metadatas=[{
                "user_id": user_id,
                **(meta or {})
            }]
        )

    def search(self, embedding, top_k=10):
        """
        Search similar embeddings.
        """
        return self.collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=top_k
        )
