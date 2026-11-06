
def recall_at_k(
    retrieved: list[str],
    relevant: list[str],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")

    relevant_set = set(relevant)

    if not relevant_set:
        return 0.0

    retrieved_set = set(retrieved[:k])

    found = retrieved_set.intersection(relevant_set)

    return len(found) / len(relevant_set)


def hit_at_k(
    retrieved: list[str],
    relevant: list[str],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")

    if not relevant:
        return 0.0

    return float(
        any(doc in set(relevant) for doc in retrieved[:k])
    )


def reciprocal_rank(
    retrieved: list[str],
    relevant: list[str],
) -> float:

    relevant_set = set(relevant)

    for rank, document in enumerate(retrieved, start=1):

        if document in relevant_set:
            return 1.0 / rank

    return 0.0
