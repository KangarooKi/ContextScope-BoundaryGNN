from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

from .data import (
    derive_boundary_labels,
    find_ego_ids,
    load_ego_network,
    stratified_split,
)
from .evaluation import aggregate_metric_rows, evaluate_binary
from .features import build_feature_views
from .interventions import suggest_interventions
from .models import (
    LogisticRegressionGD,
)


@dataclass
class ExperimentConfig:
    data_dir: str | Path
    ego_ids: list[int] | None = None
    seed: int = 7
    max_profile_dims: int = 64
    logistic_epochs: int = 180
    min_eligible_nodes: int = 12
    max_egos: int | None = None
    include_ablations: bool = False
    include_mlp: bool = False
    include_mlp_ablations: bool = False
    include_gated_mlp: bool = False
    include_node_gated_mlp: bool = False
    include_soft_labels: bool = False


def run_experiments(config: ExperimentConfig) -> dict[str, object]:
    ego_ids = config.ego_ids
    if ego_ids is None:
        ego_ids = find_ego_ids(config.data_dir)
    if config.max_egos is not None:
        ego_ids = ego_ids[: config.max_egos]
    if not ego_ids:
        raise FileNotFoundError(
            f"No SNAP `.edges` files found under {Path(config.data_dir)}"
        )

    per_ego: list[dict[str, object]] = []
    aggregate_rows: dict[str, list[dict[str, float]]] = {}
    interventions = []
    started = time.time()
    for offset, ego_id in enumerate(ego_ids):
        print(f"[{offset + 1}/{len(ego_ids)}] ego={ego_id} start", flush=True)
        ego_started = time.time()
        result = run_one_ego(config, ego_id, split_seed=config.seed + offset)
        if result is None:
            print(
                f"[{offset + 1}/{len(ego_ids)}] ego={ego_id} skipped "
                f"elapsed={time.time() - ego_started:.1f}s",
                flush=True,
            )
            continue
        per_ego.append(result)
        for name, metrics in result["metrics"].items():
            aggregate_rows.setdefault(name, []).append(metrics)
        if result.get("intervention"):
            interventions.append(result["intervention"])
        main_f1 = result["metrics"]["BoundaryGNN-Logit"]["f1"]
        print(
            f"[{offset + 1}/{len(ego_ids)}] ego={ego_id} done "
            f"nodes={result['nodes']} edges={result['edges']} "
            f"test={result['test_size']} boundary_f1={main_f1:.3f} "
            f"elapsed={time.time() - ego_started:.1f}s "
            f"total={time.time() - started:.1f}s",
            flush=True,
        )

    aggregate = {
        name: aggregate_metric_rows(rows)
        for name, rows in sorted(aggregate_rows.items())
    }
    return {
        "config": {
            "data_dir": str(config.data_dir),
            "ego_ids": ego_ids,
            "seed": config.seed,
            "max_profile_dims": config.max_profile_dims,
            "max_egos": config.max_egos,
        },
        "aggregate": aggregate,
        "per_ego": per_ego,
        "interventions": interventions,
    }


def run_one_ego(
    config: ExperimentConfig,
    ego_id: int,
    split_seed: int | None = None,
) -> dict[str, object] | None:
    data = load_ego_network(config.data_dir, ego_id)
    labels, eligible, label_info = derive_boundary_labels(data)
    eligible_count = sum(1 for flag in eligible if flag)
    positive_count = sum(labels[idx] for idx, flag in enumerate(eligible) if flag)
    if eligible_count < config.min_eligible_nodes or positive_count < 2:
        return None

    split = stratified_split(labels, eligible, seed=split_seed or config.seed)
    if not split["train"] or not split["test"]:
        return None

    views = build_feature_views(
        data,
        train_idx=split["train"],
        max_profile_dims=config.max_profile_dims,
    )
    model_specs = [
        (
            "ProfileLogit",
            LogisticRegressionGD(epochs=config.logistic_epochs, lr=0.08),
            views.profile,
        ),
        (
            "StructureLogit",
            LogisticRegressionGD(epochs=config.logistic_epochs, lr=0.08),
            views.structure,
        ),
        (
            "ProfileStructureLogit",
            LogisticRegressionGD(epochs=config.logistic_epochs, lr=0.07),
            views.profile_structure,
        ),
        (
            "BoundaryGNN-Logit",
            LogisticRegressionGD(epochs=config.logistic_epochs, lr=0.07),
            views.boundary_gnn,
        ),
    ]

    metrics: dict[str, dict[str, float]] = {}
    predictions: dict[str, list[float]] = {}
    for name, model, matrix in model_specs:
        model.fit(matrix, labels, split["train"])
        probabilities = model.predict_proba(matrix)
        predictions[name] = probabilities
        threshold = model.decision_threshold() if hasattr(model, "decision_threshold") else 0.5
        metrics[name] = evaluate_binary(labels, probabilities, split["test"], threshold)

    intervention = suggest_interventions(
        data,
        predictions["BoundaryGNN-Logit"],
        label_info,
        target_circle=None,
        top_k=5,
    )
    intervention["ego_id"] = ego_id

    return {
        "ego_id": ego_id,
        "nodes": len(data.nodes),
        "edges": len(data.edges),
        "eligible_nodes": eligible_count,
        "positive_rate": positive_count / eligible_count if eligible_count else 0.0,
        "train_size": len(split["train"]),
        "test_size": len(split["test"]),
        "relation_threshold": views.relation_threshold,
        "profile_dims": len(views.profile_dims),
        "metrics": metrics,
        "intervention": intervention,
    }
