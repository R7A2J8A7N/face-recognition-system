# test_empty_db.py

import numpy as np
from src.database import FaceDatabase
from src.matcher import FaceMatcher

# Initialize empty DB and matcher
db = FaceDatabase()
matcher = FaceMatcher()

# Fake embedding (512-d like InsightFace)
dummy_embedding = np.random.rand(512)

# Search in empty DB
results = db.search(dummy_embedding, top_k=5)

# Match result
user, score, decision = matcher.match(results)

print("User:", user)
print("Score:", score)
print("Decision:", decision)
