import math

import numpy as np


def precision_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    top_k = recommended[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for g in top_k if g in relevant)
    return hits / k


def recall_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    top_k = recommended[:k]
    if not relevant:
        return 0.0
    hits = sum(1 for g in top_k if g in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    top_k = recommended[:k]
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, g in enumerate(top_k)
        if g in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def map_at_k(recommended_lists: list[list[str]],
             relevant_sets: list[set[str]], k: int) -> float:
    scores = []
    for rec, rel in zip(recommended_lists, relevant_sets):
        ap = 0.0
        hits = 0
        for i, g in enumerate(rec[:k]):
            if g in rel:
                hits += 1
                ap += hits / (i + 1)
        scores.append(ap / min(len(rel), k) if rel else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def hit_rate_at_k(recommended_lists: list[list[str]],
                  relevant_sets: list[set[str]], k: int) -> float:
    hits = 0
    for rec, rel in zip(recommended_lists, relevant_sets):
        if any(g in rel for g in rec[:k]):
            hits += 1
    return hits / len(recommended_lists) if recommended_lists else 0.0


def coverage(recommended_lists: list[list[str]],
             total_catalog: int) -> float:
    unique_recommended = set()
    for rec in recommended_lists:
        unique_recommended.update(rec)
    return len(unique_recommended) / total_catalog if total_catalog > 0 else 0.0


def novelty(recommended_lists: list[list[str]],
            popularity: dict[str, float]) -> float:
    scores = []
    for rec in recommended_lists:
        for g in rec:
            pop = popularity.get(g, 0.0)
            if pop > 0:
                scores.append(-math.log2(pop))
    return float(np.mean(scores)) if scores else 0.0


def intra_list_similarity(recommended: list[str], recommender) -> float:
    if len(recommended) < 2:
        return 0.0
    sims = []
    game_id_to_idx = recommender.game_id_to_idx
    for i in range(len(recommended)):
        for j in range(i + 1, len(recommended)):
            ga, gb = recommended[i], recommended[j]
            if ga in game_id_to_idx and gb in game_id_to_idx:
                sim = recommender.get_similarity_between(
                    game_id_to_idx[ga], game_id_to_idx[gb]
                )
                sims.append(sim)
    return float(np.mean(sims)) if sims else 0.0


def diversity(recommended_lists: list[list[str]],
              recommender, k: int = 20) -> float:
    diversities = []
    for rec in recommended_lists[:k]:
        if len(rec) < 2:
            continue
        game_id_to_idx = recommender.game_id_to_idx
        indices = [game_id_to_idx.get(g) for g in rec if g in game_id_to_idx]
        indices = [i for i in indices if i is not None]
        if len(indices) < 2:
            continue
        pair_dissim = []
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                sim = recommender.get_similarity_between(indices[i], indices[j])
                pair_dissim.append(1.0 - sim)
        diversities.append(float(np.mean(pair_dissim)))
    return float(np.mean(diversities)) if diversities else 0.0
