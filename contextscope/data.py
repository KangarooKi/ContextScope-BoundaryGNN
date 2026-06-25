from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import tarfile
import urllib.request


SNAP_GPLUS_URL = "https://snap.stanford.edu/data/gplus.tar.gz"


@dataclass
class EgoNetwork:
    ego_id: int
    nodes: list[int]
    node_to_idx: dict[int, int]
    features: list[list[float]]
    edges: list[tuple[int, int]]
    adjacency: list[set[int]]
    circles: dict[str, set[int]]
    memberships: list[set[str]]
    is_ego: list[bool]

    def degree(self, idx: int) -> int:
        return len(self.adjacency[idx])


def download_snap_gplus(data_dir: str | Path) -> Path:
    """Download and extract SNAP's Google+ ego-network archive."""
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "gplus.tar.gz"
    if archive.exists() and not _is_valid_tar(archive):
        archive.unlink()
    if not archive.exists():
        temp_archive = Path(f"{archive}.part")
        if temp_archive.exists():
            temp_archive.unlink()
        urllib.request.urlretrieve(SNAP_GPLUS_URL, temp_archive)
        temp_archive.replace(archive)
    with tarfile.open(archive, "r:gz") as tar:
        safe_members = []
        root_resolved = root.resolve()
        for member in tar.getmembers():
            target = (root / member.name).resolve()
            if root_resolved not in [target, *target.parents]:
                raise RuntimeError(f"Unsafe tar member path: {member.name}")
            safe_members.append(member)
        tar.extractall(root, members=safe_members)
    return resolve_data_root(root)


def _is_valid_tar(path: Path) -> bool:
    try:
        return tarfile.is_tarfile(path)
    except OSError:
        return False


def resolve_data_root(data_dir: str | Path) -> Path:
    """Return the directory that contains SNAP `.edges` files."""
    root = Path(data_dir)
    if list(root.glob("*.edges")):
        return root
    for child_name in ("gplus", "facebook", "twitter"):
        nested = root / child_name
        if nested.exists() and list(nested.glob("*.edges")):
            return nested
    if root.exists():
        for nested in root.iterdir():
            if nested.is_dir() and list(nested.glob("*.edges")):
                return nested
    return root


def find_ego_ids(data_dir: str | Path) -> list[int]:
    root = resolve_data_root(data_dir)
    ego_ids = []
    for path in root.glob("*.edges"):
        try:
            ego_ids.append(int(path.stem))
        except ValueError:
            continue
    return sorted(ego_ids)


def load_ego_network(data_dir: str | Path, ego_id: int) -> EgoNetwork:
    root = resolve_data_root(data_dir)
    prefix = root / str(ego_id)
    feat_path = prefix.with_suffix(".feat")
    egofeat_path = prefix.with_suffix(".egofeat")
    edge_path = prefix.with_suffix(".edges")
    circle_path = prefix.with_suffix(".circles")

    if not edge_path.exists():
        raise FileNotFoundError(f"Missing SNAP edge file: {edge_path}")

    raw_features: dict[int, list[float]] = {}
    feature_dim = 0
    if feat_path.exists():
        with feat_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split()
                if not parts:
                    continue
                node_id = int(parts[0])
                vec = [float(v) for v in parts[1:]]
                raw_features[node_id] = vec
                feature_dim = max(feature_dim, len(vec))

    ego_feature: list[float] = []
    if egofeat_path.exists():
        text = egofeat_path.read_text(encoding="utf-8").strip()
        ego_feature = [float(v) for v in text.split()] if text else []
        feature_dim = max(feature_dim, len(ego_feature))

    raw_edges: set[tuple[int, int]] = set()
    nodes: set[int] = {ego_id}
    with edge_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            a, b = int(parts[0]), int(parts[1])
            if a == b:
                continue
            edge = (a, b) if a < b else (b, a)
            raw_edges.add(edge)
            nodes.add(a)
            nodes.add(b)

    circles: dict[str, set[int]] = {}
    if circle_path.exists():
        with circle_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                name = parts[0]
                members = {int(v) for v in parts[1:]}
                circles[name] = members
                nodes.update(members)

    nodes.update(raw_features.keys())
    ordered = sorted(nodes)
    node_to_idx = {node_id: idx for idx, node_id in enumerate(ordered)}

    def padded(vec: list[float]) -> list[float]:
        if len(vec) < feature_dim:
            return vec + [0.0] * (feature_dim - len(vec))
        return vec[:feature_dim]

    features: list[list[float]] = []
    for node_id in ordered:
        if node_id == ego_id:
            features.append(padded(ego_feature))
        else:
            features.append(padded(raw_features.get(node_id, [])))

    edges: set[tuple[int, int]] = set()
    for a, b in raw_edges:
        if a in node_to_idx and b in node_to_idx:
            ia, ib = node_to_idx[a], node_to_idx[b]
            edges.add((ia, ib) if ia < ib else (ib, ia))

    # SNAP's readme states that the ego user is connected to every alter.
    ego_idx = node_to_idx[ego_id]
    for node_id in ordered:
        if node_id == ego_id:
            continue
        idx = node_to_idx[node_id]
        edge = (ego_idx, idx) if ego_idx < idx else (idx, ego_idx)
        edges.add(edge)

    adjacency = [set() for _ in ordered]
    for ia, ib in edges:
        adjacency[ia].add(ib)
        adjacency[ib].add(ia)

    memberships = [set() for _ in ordered]
    for circle_name, members in circles.items():
        for node_id in members:
            if node_id in node_to_idx:
                memberships[node_to_idx[node_id]].add(circle_name)

    is_ego = [node_id == ego_id for node_id in ordered]
    return EgoNetwork(
        ego_id=ego_id,
        nodes=ordered,
        node_to_idx=node_to_idx,
        features=features,
        edges=sorted(edges),
        adjacency=adjacency,
        circles=circles,
        memberships=memberships,
        is_ego=is_ego,
    )


def derive_boundary_labels(
    data: EgoNetwork,
    fallback_positive_rate: float = 0.25,
) -> tuple[list[int], list[bool], dict[str, list[float]]]:
    """Create weak labels for audience-boundary risk from public circle data.

    A positive node is a friend that either belongs to multiple circles or touches
    outside-circle neighborhoods strongly enough to act as a boundary bridge.
    Nodes without any circle annotation are treated as ineligible rather than as
    reliable negatives.
    """
    n = len(data.nodes)
    labels = [0 for _ in range(n)]
    eligible = [False for _ in range(n)]
    circle_counts = [0.0 for _ in range(n)]
    neighbor_circle_diversity = [0.0 for _ in range(n)]
    outside_circle_counts = [0.0 for _ in range(n)]
    outside_neighbor_counts = [0.0 for _ in range(n)]
    outside_neighbor_ratios = [0.0 for _ in range(n)]
    bridge_scores = [0.0 for _ in range(n)]
    soft_boundary_scores = [0.0 for _ in range(n)]

    degrees = [len(neighbors) for neighbors in data.adjacency]

    for idx in range(n):
        own = data.memberships[idx]
        if data.is_ego[idx] or not own:
            continue
        eligible[idx] = True
        neighbor_circles: set[str] = set()
        outside_circles: set[str] = set()
        outside_neighbors = 0
        for nbr in data.adjacency[idx]:
            nbr_circles = data.memberships[nbr]
            neighbor_circles.update(nbr_circles)
            nbr_outside = nbr_circles - own
            outside_circles.update(nbr_outside)
            if nbr_outside:
                outside_neighbors += 1

        circle_counts[idx] = float(len(own))
        neighbor_circle_diversity[idx] = float(len(neighbor_circles))
        outside_circle_counts[idx] = float(len(outside_circles))
        outside_neighbor_counts[idx] = float(outside_neighbors)
        outside_ratio = outside_neighbors / max(1, degrees[idx])
        outside_neighbor_ratios[idx] = outside_ratio
        bridge_scores[idx] = (
            len(own)
            + 0.75 * len(outside_circles)
            + 1.50 * outside_ratio
        )

        if len(own) >= 2:
            labels[idx] = 1
        elif len(outside_circles) >= 2 and outside_ratio >= 0.25:
            labels[idx] = 1

    eligible_indices = [i for i, flag in enumerate(eligible) if flag]
    positives = sum(labels[i] for i in eligible_indices)
    min_positive = max(1, int(round(len(eligible_indices) * fallback_positive_rate)))
    if eligible_indices and positives < min_positive:
        ranked = sorted(eligible_indices, key=lambda i: bridge_scores[i], reverse=True)
        for idx in ranked[:min_positive]:
            labels[idx] = 1

    if eligible_indices:
        eligible_scores = [bridge_scores[idx] for idx in eligible_indices]
        score_min = min(eligible_scores)
        score_max = max(eligible_scores)
        score_span = max(1e-8, score_max - score_min)
        for idx in eligible_indices:
            normalized = (bridge_scores[idx] - score_min) / score_span
            if labels[idx] == 1:
                soft_boundary_scores[idx] = 0.55 + 0.45 * normalized
            else:
                soft_boundary_scores[idx] = 0.05 + 0.40 * normalized

    info = {
        "circle_counts": circle_counts,
        "neighbor_circle_diversity": neighbor_circle_diversity,
        "outside_circle_counts": outside_circle_counts,
        "outside_neighbor_counts": outside_neighbor_counts,
        "outside_neighbor_ratios": outside_neighbor_ratios,
        "bridge_scores": bridge_scores,
        "soft_boundary_scores": soft_boundary_scores,
        "degree": [float(value) for value in degrees],
    }
    return labels, eligible, info


def stratified_split(
    labels: list[int],
    eligible: list[bool],
    seed: int = 7,
    train_ratio: float = 0.60,
    valid_ratio: float = 0.20,
) -> dict[str, list[int]]:
    rng = random.Random(seed)
    pos = [idx for idx, ok in enumerate(eligible) if ok and labels[idx] == 1]
    neg = [idx for idx, ok in enumerate(eligible) if ok and labels[idx] == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)

    def split_group(items: list[int]) -> tuple[list[int], list[int], list[int]]:
        n = len(items)
        n_train = int(round(n * train_ratio))
        n_valid = int(round(n * valid_ratio))
        if n >= 3:
            n_train = min(max(1, n_train), n - 2)
            n_valid = min(max(1, n_valid), n - n_train - 1)
        return (
            items[:n_train],
            items[n_train : n_train + n_valid],
            items[n_train + n_valid :],
        )

    train_pos, valid_pos, test_pos = split_group(pos)
    train_neg, valid_neg, test_neg = split_group(neg)
    split = {
        "train": train_pos + train_neg,
        "valid": valid_pos + valid_neg,
        "test": test_pos + test_neg,
    }
    for values in split.values():
        rng.shuffle(values)
    return split
