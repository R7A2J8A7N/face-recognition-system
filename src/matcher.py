# src/matcher.py

from collections import defaultdict
from typing import Optional, Tuple


class FaceMatcher:
    """
    Advanced matcher using weighted Top-K voting.

    Handles edge cases:
    - Empty database
    - No search results
    - Borderline confidence
    """

    def __init__(
        self,
        match_threshold: float = 0.40,
        uncertain_threshold: float = 0.50,
        epsilon: float = 1e-6
    ):
        self.match_threshold = match_threshold
        self.uncertain_threshold = uncertain_threshold
        self.epsilon = epsilon

    def match(self, results) -> Tuple[Optional[str], float, str]:
        """
        Perform weighted voting on Top-K search results.

        Returns:
            user_id | None
            best_score
            decision: MATCH / UNCERTAIN / UNKNOWN
        """

        # ---------- EDGE CASE 1: No results or empty DB ----------
        if (
            not results
            or "distances" not in results
            or not results["distances"]
            or not results["distances"][0]
        ):
            return None, float("inf"), "UNKNOWN"

        user_weights = defaultdict(float)
        user_counts = defaultdict(int)

        distances = results["distances"][0]
        metadatas = results["metadatas"][0]

        # ---------- EDGE CASE 2: Metadata mismatch ----------
        if not metadatas or len(distances) != len(metadatas):
            return None, float("inf"), "UNKNOWN"

        for dist, meta in zip(distances, metadatas):
            user_id = meta.get("user_id")
            if not user_id:
                continue

            weight = 1.0 / (dist + self.epsilon)
            user_weights[user_id] += weight
            user_counts[user_id] += 1

        # ---------- EDGE CASE 3: No valid users ----------
        if not user_weights:
            return None, float("inf"), "UNKNOWN"

        # Compute final weighted score per user
        final_scores = {
            user: user_weights[user] / user_counts[user]
            for user in user_weights
        }

        best_user = max(final_scores, key=final_scores.get)
        best_weight = final_scores[best_user]
        best_distance = 1.0 / best_weight

        # ---------- Decision logic ----------
        if best_distance < self.match_threshold:
            return best_user, best_distance, "MATCH"

        if best_distance < self.uncertain_threshold:
            return best_user, best_distance, "UNCERTAIN"

        return None, best_distance, "UNKNOWN"
