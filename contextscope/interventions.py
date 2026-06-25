from __future__ import annotations

from .data import EgoNetwork


def suggest_interventions(
    data: EgoNetwork,
    probabilities: list[float],
    label_info: dict[str, list[float]],
    target_circle: str | None = None,
    top_k: int = 5,
) -> dict[str, object]:
    if not data.circles:
        return {"target_circle": None, "current_exposure": 0.0, "suggestions": []}
    circle_name = target_circle or max(data.circles, key=lambda name: len(data.circles[name]))
    member_ids = data.circles.get(circle_name, set())
    member_idx = [
        data.node_to_idx[node_id]
        for node_id in member_ids
        if node_id in data.node_to_idx
    ]
    member_set = set(member_idx)
    exposure = circle_exposure(data, probabilities, member_set)
    candidates = []
    for idx in member_idx:
        degree = len(data.adjacency[idx])
        outside_edges = sum(1 for nbr in data.adjacency[idx] if nbr not in member_set)
        outside_ratio = outside_edges / max(1, degree)
        score = probabilities[idx] * (1.0 + outside_ratio)
        candidates.append((score, idx, outside_edges, outside_ratio))
    candidates.sort(reverse=True)

    suggestions = []
    for _, idx, outside_edges, outside_ratio in candidates[:top_k]:
        risk_drop = probabilities[idx] + 0.20 * outside_edges
        suggestions.append(
            {
                "node_id": data.nodes[idx],
                "predicted_risk": probabilities[idx],
                "degree": float(len(data.adjacency[idx])),
                "outside_circle_edges": float(outside_edges),
                "outside_circle_edge_ratio": outside_ratio,
                "circle_count": label_info["circle_counts"][idx],
                "estimated_exposure_drop": risk_drop,
                "action": "review-recipient-for-this-audience-list",
            }
        )
    return {
        "target_circle": circle_name,
        "current_exposure": exposure,
        "suggestions": suggestions,
    }


def circle_exposure(
    data: EgoNetwork,
    probabilities: list[float],
    member_idx: set[int],
) -> float:
    if not member_idx:
        return 0.0
    risk_mass = sum(probabilities[idx] for idx in member_idx)
    outside_edges = 0
    for idx in member_idx:
        outside_edges += sum(1 for nbr in data.adjacency[idx] if nbr not in member_idx)
    return risk_mass + 0.20 * outside_edges

