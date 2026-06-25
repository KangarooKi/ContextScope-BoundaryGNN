from __future__ import annotations


def evaluate_binary(
    labels: list[int],
    probabilities: list[float],
    indices: list[int],
    threshold: float = 0.5,
) -> dict[str, float]:
    if not indices:
        return {
            "n": 0.0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "auc": 0.0,
        }
    tp = fp = tn = fn = 0
    y_true = []
    y_score = []
    for idx in indices:
        label = labels[idx]
        pred = 1 if probabilities[idx] >= threshold else 0
        y_true.append(label)
        y_score.append(probabilities[idx])
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 0:
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": float(len(indices)),
        "accuracy": (tp + tn) / len(indices),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": roc_auc(y_true, y_score),
    }


def roc_auc(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.0
    ranked = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum = 0.0
    rank = 1
    pos = 0
    while pos < len(ranked):
        end = pos + 1
        while end < len(ranked) and ranked[end][0] == ranked[pos][0]:
            end += 1
        avg_rank = (rank + end) / 2.0
        for _, label in ranked[pos:end]:
            if label == 1:
                rank_sum += avg_rank
        rank = end + 1
        pos = end
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def aggregate_metric_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    total_n = sum(row.get("n", 0.0) for row in rows)
    output: dict[str, float] = {"n": total_n}
    for key in ["accuracy", "precision", "recall", "f1", "auc"]:
        if total_n:
            output[key] = sum(row[key] * row.get("n", 0.0) for row in rows) / total_n
        else:
            output[key] = sum(row[key] for row in rows) / len(rows)
    return output

