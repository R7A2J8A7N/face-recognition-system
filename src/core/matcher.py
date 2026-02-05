from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Any
from src.config.settings import settings


class FaceMatcher:

    def match(
        self,
        results: List[Dict[str, Any]],
    ) -> Tuple[Optional[str], float, str]:

        if not results:
            return None, float("inf"), "UNKNOWN"

        results = results[:settings.TOP_K]

        # HARD reject floor
        best_neighbor = min(
            (r["distance"] for r in results if r.get("distance") is not None),
            default=float("inf")
        )

        if best_neighbor > settings.HARD_REJECT_THRESHOLD:
            return None, best_neighbor, "UNKNOWN"

        scores = defaultdict(float)
        counts = defaultdict(int)

        for r in results:

            user = r.get("user_id")
            dist = r.get("distance")

            if user is None or dist is None:
                continue

            if dist == float("inf") or dist != dist:
                continue

            similarity = max(0.0, 1.0 - float(dist))
            weight = 1 / (dist + settings.DISTANCE_EPSILON)

            scores[user] += similarity * weight
            counts[user] += 1

        if not scores:
            return None, float("inf"), "UNKNOWN"

        final_scores = {
            user: scores[user] / counts[user]
            for user in scores
        }

        sorted_users = sorted(
            final_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        best_user, best_similarity = sorted_users[0]

        # neighbor agreement
        if counts[best_user] < settings.MIN_VOTES:
            return None, float("inf"), "UNKNOWN"

        # margin check
        if len(sorted_users) > 1:
            second_similarity = sorted_users[1][1]

            if (best_similarity - second_similarity) < settings.MIN_SIMILARITY_MARGIN:
                return None, float("inf"), "UNCERTAIN"

        best_distance = 1.0 - best_similarity

        if best_distance < settings.MATCH_THRESHOLD:
            return best_user, best_distance, "MATCH"

        if best_distance < settings.UNCERTAIN_THRESHOLD:
            return best_user, best_distance, "UNCERTAIN"

        return None, best_distance, "UNKNOWN"
