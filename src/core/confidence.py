def distance_to_confidence(distance: float) -> float:

    if distance < 0.3:
        return 0.98
    if distance < 0.4:
        return 0.92
    if distance < 0.5:
        return 0.80

    return 0.0
