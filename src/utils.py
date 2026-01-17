# src/utils.py
def distance_to_confidence(distance):
    """
    Convert cosine distance to confidence score.
    """
    return max(0.0, 1.0 - distance)
