from collections import defaultdict
from typing import List, Dict, Optional, Tuple
from src.config.settings import settings


class FaceMatcher:

    def match(
        self,
        results: List[Dict],
    ) -> Tuple[Optional[str], float, str]:

        if not results:
            return None, float("inf"), "UNKNOWN"

        weights = defaultdict(float)
        counts = defaultdict(int)

        for r in results:

            user = r["user_id"]
            dist = r["distance"]

            weight = 1.0 / (dist + 1e-6)

            weights[user] += weight
            counts[user] += 1

        final_scores = {
            u: weights[u] / counts[u]
            for u in weights
        }

        best_user = max(final_scores, key=lambda k: final_scores[k])

        best_distance = 1.0 / final_scores[best_user]

        if best_distance < settings.MATCH_THRESHOLD:
            return best_user, best_distance, "MATCH"

        if best_distance < settings.UNCERTAIN_THRESHOLD:
            return best_user, best_distance, "UNCERTAIN"

        return None, best_distance, "UNKNOWN"
