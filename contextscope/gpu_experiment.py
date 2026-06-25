from __future__ import annotations

from pathlib import Path
import math
import time

import torch

from .data import (
    derive_boundary_labels,
    find_ego_ids,
    load_ego_network,
    stratified_split,
)
from .evaluation import aggregate_metric_rows, evaluate_binary
from .experiment import ExperimentConfig
from .features import clustering_coefficient, project_columns, select_profile_dimensions
from .interventions import suggest_interventions


OUR_METHOD = "BoundaryGNN-Logit"
SOFT_LOGIT_METHOD = "BoundaryGNN-SoftLogit"
CHANNEL_NAMES = ["self", "all", "similar", "dissimilar", "structure"]
ABLATION_SPECS = [
    ("BoundaryGNN-w/o-All", "boundary_no_all"),
    ("BoundaryGNN-w/o-Similar", "boundary_no_similar"),
    ("BoundaryGNN-w/o-Dissimilar", "boundary_no_dissimilar"),
    ("BoundaryGNN-w/o-Structure", "boundary_no_structure"),
    ("BoundaryGNN-w/o-EdgeTyping", "boundary_no_edge_typing"),
]
MLP_HEAD_SPECS = [
    ("BoundaryGNN-MLP-128", [128], 0.10, 0.0030),
    ("BoundaryGNN-MLP-512", [512], 0.15, 0.0020),
    ("BoundaryGNN-DeepMLP-1024x512", [1024, 512], 0.20, 0.0015),
    ("BoundaryGNN-DeepMLP-2048x1024", [2048, 1024], 0.25, 0.0010),
]
MLP_ABLATION_HEAD = MLP_HEAD_SPECS[-1]
GATED_MLP_HEAD = (
    "BoundaryGNN-GatedDeepMLP-2048x1024",
    [2048, 1024],
    0.25,
    0.0010,
)
NODE_GATED_MLP_HEAD = (
    "BoundaryGNN-NodeGatedDeepMLP-2048x1024",
    GATED_MLP_HEAD[1],
    GATED_MLP_HEAD[2],
    GATED_MLP_HEAD[3],
    128,
)
SOFT_MLP_HEAD = (
    "BoundaryGNN-SoftDeepMLP-2048x1024",
    MLP_ABLATION_HEAD[1],
    MLP_ABLATION_HEAD[2],
    MLP_ABLATION_HEAD[3],
)
SOFT_GATED_MLP_HEAD = (
    "BoundaryGNN-GatedSoftDeepMLP-2048x1024",
    GATED_MLP_HEAD[1],
    GATED_MLP_HEAD[2],
    GATED_MLP_HEAD[3],
)


def run_gpu_experiments(config: ExperimentConfig, device_name: str = "cuda") -> dict[str, object]:
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    device = torch.device(device_name)

    ego_ids = config.ego_ids if config.ego_ids is not None else find_ego_ids(config.data_dir)
    if config.max_egos is not None:
        ego_ids = ego_ids[: config.max_egos]
    if not ego_ids:
        raise FileNotFoundError(f"No SNAP `.edges` files found under {Path(config.data_dir)}")

    per_ego: list[dict[str, object]] = []
    aggregate_rows: dict[str, list[dict[str, float]]] = {}
    model_info_rows: dict[str, list[dict[str, float]]] = {}
    interventions = []
    started = time.time()
    for offset, ego_id in enumerate(ego_ids):
        print(f"[{offset + 1}/{len(ego_ids)}] ego={ego_id} start device={device}", flush=True)
        ego_started = time.time()
        result = run_one_ego_gpu(config, ego_id, device, split_seed=config.seed + offset)
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
        for name, info in result.get("model_info", {}).items():
            model_info_rows.setdefault(name, []).append(info)
        if result.get("intervention"):
            interventions.append(result["intervention"])
        main_f1 = result["metrics"][OUR_METHOD]["f1"]
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
    model_info = {
        name: aggregate_model_info(rows)
        for name, rows in sorted(model_info_rows.items())
    }
    return {
        "config": {
            "data_dir": str(config.data_dir),
            "ego_ids": ego_ids,
            "seed": config.seed,
            "max_profile_dims": config.max_profile_dims,
            "max_egos": config.max_egos,
            "device": str(device),
            "include_ablations": config.include_ablations,
            "include_mlp": config.include_mlp,
            "include_mlp_ablations": config.include_mlp_ablations,
            "include_gated_mlp": config.include_gated_mlp,
            "include_node_gated_mlp": config.include_node_gated_mlp,
            "include_soft_labels": config.include_soft_labels,
        },
        "aggregate": aggregate,
        "model_info": model_info,
        "per_ego": per_ego,
        "interventions": interventions,
    }


def run_one_ego_gpu(
    config: ExperimentConfig,
    ego_id: int,
    device: torch.device,
    split_seed: int,
) -> dict[str, object] | None:
    data = load_ego_network(config.data_dir, ego_id)
    labels, eligible, label_info = derive_boundary_labels(data)
    eligible_count = sum(1 for flag in eligible if flag)
    positive_count = sum(labels[idx] for idx, flag in enumerate(eligible) if flag)
    if eligible_count < config.min_eligible_nodes or positive_count < 2:
        return None

    split = stratified_split(labels, eligible, seed=split_seed)
    if not split["train"] or not split["test"]:
        return None

    views = build_torch_feature_views(data, split["train"], config.max_profile_dims, device)
    y = torch.tensor(labels, dtype=torch.float32, device=device)
    soft_targets = label_info.get("soft_boundary_scores", [float(value) for value in labels])
    y_soft = torch.tensor(soft_targets, dtype=torch.float32, device=device)

    logit_specs = [
        ("ProfileLogit", views["profile"]),
        ("StructureLogit", views["structure"]),
        ("ProfileStructureLogit", views["profile_structure"]),
    ]
    if config.include_ablations:
        logit_specs.extend((name, views[key]) for name, key in ABLATION_SPECS)

    metrics: dict[str, dict[str, float]] = {}
    predictions: dict[str, list[float]] = {}
    model_info: dict[str, dict[str, float]] = {}

    def fit_logit(name: str, matrix: torch.Tensor, targets: torch.Tensor = y) -> None:
        probabilities = train_torch_logit(
            matrix,
            targets,
            split["train"],
            epochs=config.logistic_epochs,
            lr=0.06,
            l2=1e-4,
        )
        prob_list = probabilities.detach().cpu().tolist()
        predictions[name] = prob_list
        threshold = best_f1_threshold(labels, prob_list, split["valid"])
        metrics[name] = evaluate_binary(labels, prob_list, split["test"], threshold)
        model_info[name] = {
            "feature_dim": float(matrix.shape[1]),
            "parameter_count": float(linear_parameter_count(matrix.shape[1])),
        }

    for name, matrix in logit_specs:
        fit_logit(name, matrix)
    fit_logit(OUR_METHOD, views["boundary_gnn"])
    if config.include_soft_labels:
        fit_logit(SOFT_LOGIT_METHOD, views["boundary_gnn"], y_soft)

    def fit_mlp(
        name: str,
        matrix: torch.Tensor,
        hidden_dims: list[int],
        dropout: float,
        lr: float,
        seed_offset: int,
        targets: torch.Tensor = y,
    ) -> None:
        probabilities = train_torch_mlp(
            matrix,
            targets,
            split["train"],
            epochs=config.logistic_epochs,
            lr=lr,
            l2=5e-5,
            hidden_dims=hidden_dims,
            dropout=dropout,
            seed=split_seed + seed_offset,
        )
        prob_list = probabilities.detach().cpu().tolist()
        predictions[name] = prob_list
        threshold = best_f1_threshold(labels, prob_list, split["valid"])
        metrics[name] = evaluate_binary(labels, prob_list, split["test"], threshold)
        model_info[name] = {
            "feature_dim": float(matrix.shape[1]),
            "parameter_count": float(mlp_parameter_count(matrix.shape[1], hidden_dims)),
        }

    if config.include_mlp:
        for name, hidden_dims, dropout, lr in MLP_HEAD_SPECS:
            fit_mlp(
                name,
                views["boundary_gnn"],
                hidden_dims,
                dropout,
                lr,
                seed_offset=sum(hidden_dims),
            )

    if config.include_mlp_ablations:
        base_name, hidden_dims, dropout, lr = MLP_ABLATION_HEAD
        if base_name not in metrics:
            fit_mlp(
                base_name,
                views["boundary_gnn"],
                hidden_dims,
                dropout,
                lr,
                seed_offset=sum(hidden_dims),
            )
        for logit_name, view_key in ABLATION_SPECS:
            suffix = logit_name.replace("BoundaryGNN-", "")
            fit_mlp(
                f"{base_name}-{suffix}",
                views[view_key],
                hidden_dims,
                dropout,
                lr,
                seed_offset=sum(hidden_dims) + len(suffix),
            )

    if config.include_soft_labels:
        soft_name, soft_hidden_dims, soft_dropout, soft_lr = SOFT_MLP_HEAD
        fit_mlp(
            soft_name,
            views["boundary_gnn"],
            soft_hidden_dims,
            soft_dropout,
            soft_lr,
            seed_offset=sum(soft_hidden_dims) + 23,
            targets=y_soft,
        )

    if config.include_gated_mlp or config.include_node_gated_mlp:
        base_name, hidden_dims, dropout, lr = MLP_ABLATION_HEAD
        if base_name not in metrics:
            fit_mlp(
                base_name,
                views["boundary_gnn"],
                hidden_dims,
                dropout,
                lr,
                seed_offset=sum(hidden_dims),
            )
        channels = [views["boundary_channels"][name] for name in CHANNEL_NAMES]

    if config.include_gated_mlp:
        gated_name, gated_hidden_dims, gated_dropout, gated_lr = GATED_MLP_HEAD
        probabilities, gates = train_torch_gated_mlp(
            channels,
            y,
            split["train"],
            epochs=config.logistic_epochs,
            lr=gated_lr,
            l2=5e-5,
            hidden_dims=gated_hidden_dims,
            dropout=gated_dropout,
            seed=split_seed + sum(gated_hidden_dims) + 17,
        )
        prob_list = probabilities.detach().cpu().tolist()
        predictions[gated_name] = prob_list
        threshold = best_f1_threshold(labels, prob_list, split["valid"])
        metrics[gated_name] = evaluate_binary(labels, prob_list, split["test"], threshold)
        info = {
            "feature_dim": float(sum(channel.shape[1] for channel in channels)),
            "parameter_count": float(
                gated_mlp_parameter_count([channel.shape[1] for channel in channels], gated_hidden_dims)
            ),
        }
        for channel_name, gate_value in zip(CHANNEL_NAMES, gates):
            info[f"gate_{channel_name}"] = float(gate_value)
        model_info[gated_name] = info

        if config.include_soft_labels:
            soft_gated_name, soft_hidden_dims, soft_dropout, soft_lr = SOFT_GATED_MLP_HEAD
            probabilities, gates = train_torch_gated_mlp(
                channels,
                y_soft,
                split["train"],
                epochs=config.logistic_epochs,
                lr=soft_lr,
                l2=5e-5,
                hidden_dims=soft_hidden_dims,
                dropout=soft_dropout,
                seed=split_seed + sum(soft_hidden_dims) + 41,
            )
            prob_list = probabilities.detach().cpu().tolist()
            predictions[soft_gated_name] = prob_list
            threshold = best_f1_threshold(labels, prob_list, split["valid"])
            metrics[soft_gated_name] = evaluate_binary(labels, prob_list, split["test"], threshold)
            info = {
                "feature_dim": float(sum(channel.shape[1] for channel in channels)),
                "parameter_count": float(
                    gated_mlp_parameter_count(
                        [channel.shape[1] for channel in channels],
                        soft_hidden_dims,
                    )
                ),
            }
            for channel_name, gate_value in zip(CHANNEL_NAMES, gates):
                info[f"gate_{channel_name}"] = float(gate_value)
            model_info[soft_gated_name] = info

    if config.include_node_gated_mlp:
        node_name, node_hidden_dims, node_dropout, node_lr, gate_hidden_dim = NODE_GATED_MLP_HEAD
        probabilities, gates = train_torch_node_gated_mlp(
            channels,
            y,
            split["train"],
            epochs=config.logistic_epochs,
            lr=node_lr,
            l2=5e-5,
            hidden_dims=node_hidden_dims,
            dropout=node_dropout,
            gate_hidden_dim=gate_hidden_dim,
            seed=split_seed + sum(node_hidden_dims) + gate_hidden_dim + 29,
        )
        prob_list = probabilities.detach().cpu().tolist()
        predictions[node_name] = prob_list
        threshold = best_f1_threshold(labels, prob_list, split["valid"])
        metrics[node_name] = evaluate_binary(labels, prob_list, split["test"], threshold)
        info = {
            "feature_dim": float(sum(channel.shape[1] for channel in channels)),
            "parameter_count": float(
                node_gated_mlp_parameter_count(
                    [channel.shape[1] for channel in channels],
                    node_hidden_dims,
                    gate_hidden_dim,
                )
            ),
            "gate_hidden_dim": float(gate_hidden_dim),
        }
        for channel_name, gate_value in zip(CHANNEL_NAMES, gates):
            info[f"gate_{channel_name}"] = float(gate_value)
        model_info[node_name] = info

    intervention = suggest_interventions(
        data,
        predictions[OUR_METHOD],
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
        "relation_threshold": views["relation_threshold"],
        "profile_dims": views["profile_dims"],
        "metrics": metrics,
        "model_info": model_info,
        "intervention": intervention,
    }


def build_torch_feature_views(
    data,
    train_idx: list[int],
    max_profile_dims: int,
    device: torch.device,
) -> dict[str, object]:
    dims = select_profile_dimensions(data.features, train_idx, max_profile_dims)
    raw_profile = project_columns(data.features, dims)
    raw = torch.tensor(raw_profile, dtype=torch.float32, device=device)
    profile = zscore_tensor(raw, train_idx)

    edge_index = torch.tensor(data.edges, dtype=torch.long, device=device)
    if edge_index.numel() == 0:
        src = torch.empty(0, dtype=torch.long, device=device)
        dst = torch.empty(0, dtype=torch.long, device=device)
    else:
        src = edge_index[:, 0]
        dst = edge_index[:, 1]

    similar_mask, threshold = edge_similarity_mask(raw, src, dst)
    all_neighbor, degree = aggregate_edges(profile, src, dst)
    gcn_neighbor = aggregate_gcn(profile, src, dst, degree)
    similar_neighbor, similar_degree = aggregate_edges(profile, src, dst, similar_mask)
    dissimilar_neighbor, dissimilar_degree = aggregate_edges(profile, src, dst, ~similar_mask)

    structure = build_structure_tensor(
        data,
        device,
        degree,
        similar_degree,
        dissimilar_degree,
        src,
        dst,
    )
    structure = zscore_tensor(structure, train_idx)

    gcn_gnn = zscore_tensor(torch.cat([profile, gcn_neighbor, structure], dim=1), train_idx)
    profile_structure = zscore_tensor(torch.cat([profile, structure], dim=1), train_idx)
    boundary_gnn = zscore_tensor(
        torch.cat([profile, all_neighbor, similar_neighbor, dissimilar_neighbor, structure], dim=1),
        train_idx,
    )
    channel_widths = [
        profile.shape[1],
        all_neighbor.shape[1],
        similar_neighbor.shape[1],
        dissimilar_neighbor.shape[1],
        structure.shape[1],
    ]
    boundary_channel_tensors = torch.split(boundary_gnn, channel_widths, dim=1)
    boundary_no_all = zscore_tensor(
        torch.cat([profile, similar_neighbor, dissimilar_neighbor, structure], dim=1),
        train_idx,
    )
    boundary_no_similar = zscore_tensor(
        torch.cat([profile, all_neighbor, dissimilar_neighbor, structure], dim=1),
        train_idx,
    )
    boundary_no_dissimilar = zscore_tensor(
        torch.cat([profile, all_neighbor, similar_neighbor, structure], dim=1),
        train_idx,
    )
    boundary_no_structure = zscore_tensor(
        torch.cat([profile, all_neighbor, similar_neighbor, dissimilar_neighbor], dim=1),
        train_idx,
    )
    boundary_no_edge_typing = zscore_tensor(
        torch.cat([profile, all_neighbor, structure], dim=1),
        train_idx,
    )

    return {
        "profile": profile,
        "structure": structure,
        "profile_structure": profile_structure,
        "gcn_gnn": gcn_gnn,
        "boundary_gnn": boundary_gnn,
        "boundary_no_all": boundary_no_all,
        "boundary_no_similar": boundary_no_similar,
        "boundary_no_dissimilar": boundary_no_dissimilar,
        "boundary_no_structure": boundary_no_structure,
        "boundary_no_edge_typing": boundary_no_edge_typing,
        "boundary_channels": {
            name: channel
            for name, channel in zip(CHANNEL_NAMES, boundary_channel_tensors)
        },
        "all_neighbor": all_neighbor,
        "similar_neighbor": similar_neighbor,
        "dissimilar_neighbor": dissimilar_neighbor,
        "relation_threshold": float(threshold),
        "profile_dims": len(dims),
    }


def edge_similarity_mask(
    raw_profile: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    sample_size: int = 50000,
    batch_size: int = 250000,
) -> tuple[torch.Tensor, float]:
    if src.numel() == 0:
        return torch.empty(0, dtype=torch.bool, device=raw_profile.device), 0.0
    binary = raw_profile > 0.0
    step = max(1, src.numel() // sample_size)
    sample_pos = torch.arange(0, src.numel(), step, device=raw_profile.device)[:sample_size]
    sample_sim = jaccard_batch(binary, src[sample_pos], dst[sample_pos])
    threshold = torch.median(sample_sim)
    positive = sample_sim[sample_sim > 0.0]
    if threshold.item() <= 0.0 and positive.numel() > 0:
        threshold = torch.median(positive)

    output = torch.empty(src.numel(), dtype=torch.bool, device=raw_profile.device)
    for start in range(0, src.numel(), batch_size):
        end = min(start + batch_size, src.numel())
        sim = jaccard_batch(binary, src[start:end], dst[start:end])
        output[start:end] = (sim >= threshold) & (sim > 0.0)
    return output, float(threshold.item())


def jaccard_batch(binary: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    left = binary[src]
    right = binary[dst]
    intersection = (left & right).sum(dim=1).float()
    union = (left | right).sum(dim=1).float().clamp_min(1.0)
    return intersection / union


def aggregate_edges(
    features: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = features.shape[0]
    output = torch.zeros_like(features)
    counts = torch.zeros(rows, dtype=features.dtype, device=features.device)
    if mask is not None:
        src = src[mask]
        dst = dst[mask]
    if src.numel() == 0:
        return output, counts
    output.index_add_(0, src, features[dst])
    output.index_add_(0, dst, features[src])
    ones = torch.ones(src.shape[0], dtype=features.dtype, device=features.device)
    counts.index_add_(0, src, ones)
    counts.index_add_(0, dst, ones)
    output = output / counts.clamp_min(1.0).unsqueeze(1)
    return output, counts


def aggregate_gcn(
    features: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    degree_without_self: torch.Tensor,
) -> torch.Tensor:
    degree = degree_without_self + 1.0
    output = features / degree.clamp_min(1.0).unsqueeze(1)
    if src.numel() == 0:
        return output
    scale = torch.rsqrt(degree[src] * degree[dst])
    output.index_add_(0, src, features[dst] * scale.unsqueeze(1))
    output.index_add_(0, dst, features[src] * scale.unsqueeze(1))
    return output


def build_structure_tensor(
    data,
    device: torch.device,
    degree: torch.Tensor,
    similar_degree: torch.Tensor,
    dissimilar_degree: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
) -> torch.Tensor:
    rows = len(data.nodes)
    degree_safe = degree.clamp_min(1.0)
    max_degree = degree.max().clamp_min(1.0)
    neighbor_degree_sum = torch.zeros(rows, dtype=torch.float32, device=device)
    if src.numel() > 0:
        neighbor_degree_sum.index_add_(0, src, degree[dst])
        neighbor_degree_sum.index_add_(0, dst, degree[src])
    avg_neighbor_degree = neighbor_degree_sum / degree_safe
    clustering = torch.tensor(
        [clustering_coefficient(data.adjacency, idx) for idx in range(rows)],
        dtype=torch.float32,
        device=device,
    )
    is_ego = torch.tensor(data.is_ego, dtype=torch.float32, device=device)
    return torch.stack(
        [
            torch.log1p(degree),
            degree / max_degree,
            clustering,
            torch.log1p(avg_neighbor_degree),
            similar_degree / degree_safe,
            dissimilar_degree / degree_safe,
            is_ego,
        ],
        dim=1,
    )


def zscore_tensor(matrix: torch.Tensor, train_idx: list[int]) -> torch.Tensor:
    if matrix.numel() == 0:
        return matrix
    idx = torch.tensor(train_idx, dtype=torch.long, device=matrix.device)
    train = matrix[idx]
    mean = train.mean(dim=0)
    scale = train.std(dim=0, unbiased=False).clamp_min(1e-8)
    return (matrix - mean) / scale


def train_torch_logit(
    matrix: torch.Tensor,
    labels: torch.Tensor,
    train_idx: list[int],
    epochs: int,
    lr: float,
    l2: float,
) -> torch.Tensor:
    idx = torch.tensor(train_idx, dtype=torch.long, device=matrix.device)
    x_train = matrix[idx]
    y_train = labels[idx]
    weights = torch.zeros(matrix.shape[1], dtype=torch.float32, device=matrix.device, requires_grad=True)
    pos_rate = ((y_train.sum() + 0.5) / (y_train.numel() + 1.0)).clamp(1e-5, 1.0 - 1e-5)
    bias = torch.tensor(
        math.log(float(pos_rate / (1.0 - pos_rate))),
        dtype=torch.float32,
        device=matrix.device,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam([weights, bias], lr=lr)
    positives = y_train.sum().clamp_min(1.0)
    negatives = (y_train.numel() - y_train.sum()).clamp_min(1.0)
    pos_weight = (negatives / positives).clamp(0.25, 4.0)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = x_train @ weights + bias
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            y_train,
            pos_weight=pos_weight,
        )
        loss = loss + 0.5 * l2 * (weights @ weights)
        loss.backward()
        optimizer.step()
    return torch.sigmoid(matrix @ weights.detach() + bias.detach())


def train_torch_mlp(
    matrix: torch.Tensor,
    labels: torch.Tensor,
    train_idx: list[int],
    epochs: int,
    lr: float,
    l2: float,
    hidden_dims: list[int],
    dropout: float,
    seed: int,
) -> torch.Tensor:
    torch.manual_seed(seed)
    idx = torch.tensor(train_idx, dtype=torch.long, device=matrix.device)
    x_train = matrix[idx]
    y_train = labels[idx]
    layers: list[torch.nn.Module] = []
    width = matrix.shape[1]
    for hidden_dim in hidden_dims:
        layers.extend(
            [
                torch.nn.Linear(width, hidden_dim),
                torch.nn.LayerNorm(hidden_dim),
                torch.nn.ReLU(),
            ]
        )
        if dropout > 0.0:
            layers.append(torch.nn.Dropout(dropout))
        width = hidden_dim
    layers.append(torch.nn.Linear(width, 1))
    model = torch.nn.Sequential(*layers).to(matrix.device)
    pos_rate = ((y_train.sum() + 0.5) / (y_train.numel() + 1.0)).clamp(1e-5, 1.0 - 1e-5)
    with torch.no_grad():
        final = model[-1]
        assert isinstance(final, torch.nn.Linear)
        final.bias.fill_(math.log(float(pos_rate / (1.0 - pos_rate))))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    positives = y_train.sum().clamp_min(1.0)
    negatives = (y_train.numel() - y_train.sum()).clamp_min(1.0)
    pos_weight = (negatives / positives).clamp(0.25, 4.0)
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_train).squeeze(1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            y_train,
            pos_weight=pos_weight,
        )
        penalty = torch.zeros((), dtype=torch.float32, device=matrix.device)
        for parameter in model.parameters():
            penalty = penalty + parameter.square().sum()
        loss = loss + 0.5 * l2 * penalty
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        model.eval()
        return torch.sigmoid(model(matrix).squeeze(1))


class ChannelGatedMLP(torch.nn.Module):
    def __init__(
        self,
        channel_dims: list[int],
        hidden_dims: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.raw_gates = torch.nn.Parameter(torch.zeros(len(channel_dims)))
        layers: list[torch.nn.Module] = []
        width = sum(channel_dims)
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    torch.nn.Linear(width, hidden_dim),
                    torch.nn.LayerNorm(hidden_dim),
                    torch.nn.ReLU(),
                ]
            )
            if dropout > 0.0:
                layers.append(torch.nn.Dropout(dropout))
            width = hidden_dim
        layers.append(torch.nn.Linear(width, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def gates(self) -> torch.Tensor:
        return 2.0 * torch.sigmoid(self.raw_gates)

    def forward(self, channels: list[torch.Tensor]) -> torch.Tensor:
        gates = self.gates()
        scaled = [channel * gates[pos] for pos, channel in enumerate(channels)]
        return self.mlp(torch.cat(scaled, dim=1)).squeeze(1)


class NodeGatedMLP(torch.nn.Module):
    def __init__(
        self,
        channel_dims: list[int],
        hidden_dims: list[int],
        dropout: float,
        gate_hidden_dim: int,
    ) -> None:
        super().__init__()
        self.channel_count = len(channel_dims)
        input_dim = sum(channel_dims)
        self.gate_input = torch.nn.Sequential(
            torch.nn.Linear(input_dim, gate_hidden_dim),
            torch.nn.LayerNorm(gate_hidden_dim),
            torch.nn.ReLU(),
        )
        self.gate_output = torch.nn.Linear(gate_hidden_dim, self.channel_count)
        torch.nn.init.zeros_(self.gate_output.weight)
        torch.nn.init.zeros_(self.gate_output.bias)

        layers: list[torch.nn.Module] = []
        width = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    torch.nn.Linear(width, hidden_dim),
                    torch.nn.LayerNorm(hidden_dim),
                    torch.nn.ReLU(),
                ]
            )
            if dropout > 0.0:
                layers.append(torch.nn.Dropout(dropout))
            width = hidden_dim
        layers.append(torch.nn.Linear(width, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def gates(self, channels: list[torch.Tensor]) -> torch.Tensor:
        joined = torch.cat(channels, dim=1)
        logits = self.gate_output(self.gate_input(joined))
        return 2.0 * torch.sigmoid(logits)

    def forward(self, channels: list[torch.Tensor]) -> torch.Tensor:
        gates = self.gates(channels)
        scaled = [
            channel * gates[:, pos].unsqueeze(1)
            for pos, channel in enumerate(channels)
        ]
        return self.mlp(torch.cat(scaled, dim=1)).squeeze(1)


def train_torch_gated_mlp(
    channels: list[torch.Tensor],
    labels: torch.Tensor,
    train_idx: list[int],
    epochs: int,
    lr: float,
    l2: float,
    hidden_dims: list[int],
    dropout: float,
    seed: int,
) -> tuple[torch.Tensor, list[float]]:
    torch.manual_seed(seed)
    idx = torch.tensor(train_idx, dtype=torch.long, device=labels.device)
    train_channels = [channel[idx] for channel in channels]
    y_train = labels[idx]
    model = ChannelGatedMLP(
        [channel.shape[1] for channel in channels],
        hidden_dims,
        dropout,
    ).to(labels.device)
    pos_rate = ((y_train.sum() + 0.5) / (y_train.numel() + 1.0)).clamp(1e-5, 1.0 - 1e-5)
    with torch.no_grad():
        final = model.mlp[-1]
        assert isinstance(final, torch.nn.Linear)
        final.bias.fill_(math.log(float(pos_rate / (1.0 - pos_rate))))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    positives = y_train.sum().clamp_min(1.0)
    negatives = (y_train.numel() - y_train.sum()).clamp_min(1.0)
    pos_weight = (negatives / positives).clamp(0.25, 4.0)
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_channels)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            y_train,
            pos_weight=pos_weight,
        )
        penalty = torch.zeros((), dtype=torch.float32, device=labels.device)
        for parameter in model.parameters():
            penalty = penalty + parameter.square().sum()
        loss = loss + 0.5 * l2 * penalty
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        model.eval()
        probabilities = torch.sigmoid(model(channels))
        gates = model.gates().detach().cpu().tolist()
    return probabilities, gates


def train_torch_node_gated_mlp(
    channels: list[torch.Tensor],
    labels: torch.Tensor,
    train_idx: list[int],
    epochs: int,
    lr: float,
    l2: float,
    hidden_dims: list[int],
    dropout: float,
    gate_hidden_dim: int,
    seed: int,
) -> tuple[torch.Tensor, list[float]]:
    torch.manual_seed(seed)
    idx = torch.tensor(train_idx, dtype=torch.long, device=labels.device)
    train_channels = [channel[idx] for channel in channels]
    y_train = labels[idx]
    model = NodeGatedMLP(
        [channel.shape[1] for channel in channels],
        hidden_dims,
        dropout,
        gate_hidden_dim,
    ).to(labels.device)
    pos_rate = ((y_train.sum() + 0.5) / (y_train.numel() + 1.0)).clamp(1e-5, 1.0 - 1e-5)
    with torch.no_grad():
        final = model.mlp[-1]
        assert isinstance(final, torch.nn.Linear)
        final.bias.fill_(math.log(float(pos_rate / (1.0 - pos_rate))))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    positives = y_train.sum().clamp_min(1.0)
    negatives = (y_train.numel() - y_train.sum()).clamp_min(1.0)
    pos_weight = (negatives / positives).clamp(0.25, 4.0)
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_channels)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            y_train,
            pos_weight=pos_weight,
        )
        penalty = torch.zeros((), dtype=torch.float32, device=labels.device)
        for parameter in model.parameters():
            penalty = penalty + parameter.square().sum()
        loss = loss + 0.5 * l2 * penalty
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        model.eval()
        probabilities = torch.sigmoid(model(channels))
        gates = model.gates(channels).mean(dim=0).detach().cpu().tolist()
    return probabilities, gates


def linear_parameter_count(feature_dim: int) -> int:
    return feature_dim + 1


def mlp_parameter_count(feature_dim: int, hidden_dims: list[int]) -> int:
    total = 0
    width = feature_dim
    for hidden_dim in hidden_dims:
        total += (width + 1) * hidden_dim
        total += 2 * hidden_dim
        width = hidden_dim
    total += width + 1
    return total


def gated_mlp_parameter_count(channel_dims: list[int], hidden_dims: list[int]) -> int:
    return len(channel_dims) + mlp_parameter_count(sum(channel_dims), hidden_dims)


def node_gated_mlp_parameter_count(
    channel_dims: list[int],
    hidden_dims: list[int],
    gate_hidden_dim: int,
) -> int:
    input_dim = sum(channel_dims)
    channel_count = len(channel_dims)
    gate_parameters = (input_dim + 1) * gate_hidden_dim
    gate_parameters += 2 * gate_hidden_dim
    gate_parameters += (gate_hidden_dim + 1) * channel_count
    return gate_parameters + mlp_parameter_count(input_dim, hidden_dims)


def aggregate_model_info(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    output = {}
    for key in [
        "feature_dim",
        "parameter_count",
        "gate_hidden_dim",
        *(f"gate_{name}" for name in CHANNEL_NAMES),
    ]:
        values = [row[key] for row in rows if key in row]
        if not values:
            continue
        output[f"{key}_mean"] = sum(values) / len(values)
        output[f"{key}_min"] = min(values)
        output[f"{key}_max"] = max(values)
    return output


def best_f1_threshold(labels: list[int], probabilities: list[float], indices: list[int]) -> float:
    if not indices:
        return 0.5
    candidates = sorted({probabilities[idx] for idx in indices})
    if not candidates:
        return 0.5
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in candidates:
        row = evaluate_binary(labels, probabilities, indices, threshold)
        if row["f1"] > best_f1:
            best_f1 = row["f1"]
            best_threshold = threshold
    return best_threshold
