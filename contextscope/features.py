from __future__ import annotations

from dataclasses import dataclass
import math

from .data import EgoNetwork

try:
    import numpy as np
except ImportError:
    np = None


@dataclass
class FeatureViews:
    profile: list[list[float]]
    structure: list[list[float]]
    profile_structure: list[list[float]]
    gcn_gnn: list[list[float]]
    boundary_gnn: list[list[float]]
    profile_dims: list[int]
    relation_threshold: float
    similar_adjacency: list[set[int]]
    dissimilar_adjacency: list[set[int]]


def build_feature_views(
    data: EgoNetwork,
    train_idx: list[int],
    max_profile_dims: int = 64,
) -> FeatureViews:
    dims = select_profile_dimensions(data.features, train_idx, max_profile_dims)
    raw_profile = project_columns(data.features, dims)
    profile = zscore(raw_profile, train_idx)

    similar_adj, dissimilar_adj, threshold = split_edges_by_profile_similarity(
        data, raw_profile
    )
    all_neighbor = mean_neighbor_features(profile, data.adjacency)
    gcn_neighbor = gcn_normalized_features(profile, data.adjacency)
    similar_neighbor = mean_neighbor_features(profile, similar_adj)
    dissimilar_neighbor = mean_neighbor_features(profile, dissimilar_adj)

    structure_raw = structural_features(data, similar_adj, dissimilar_adj)
    structure = zscore(structure_raw, train_idx)

    profile_structure_raw = hstack([profile, structure])
    gcn_gnn_raw = hstack([profile, gcn_neighbor, structure])
    boundary_gnn_raw = hstack(
        [profile, all_neighbor, similar_neighbor, dissimilar_neighbor, structure]
    )
    return FeatureViews(
        profile=profile,
        structure=structure,
        profile_structure=zscore(profile_structure_raw, train_idx),
        gcn_gnn=zscore(gcn_gnn_raw, train_idx),
        boundary_gnn=zscore(boundary_gnn_raw, train_idx),
        profile_dims=dims,
        relation_threshold=threshold,
        similar_adjacency=similar_adj,
        dissimilar_adjacency=dissimilar_adj,
    )


def select_profile_dimensions(
    matrix: list[list[float]],
    train_idx: list[int],
    max_dims: int,
) -> list[int]:
    if not matrix or not matrix[0] or max_dims <= 0:
        return []
    width = len(matrix[0])
    if width <= max_dims:
        return list(range(width))
    rows = train_idx or list(range(len(matrix)))
    scored: list[tuple[float, int]] = []
    for col in range(width):
        values = [matrix[row][col] for row in rows]
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        scored.append((variance, col))
    scored.sort(reverse=True)
    return sorted(col for _, col in scored[:max_dims])


def project_columns(matrix: list[list[float]], dims: list[int]) -> list[list[float]]:
    if not dims:
        return [[0.0] for _ in matrix]
    return [[row[col] if col < len(row) else 0.0 for col in dims] for row in matrix]


def split_edges_by_profile_similarity(
    data: EgoNetwork,
    profile: list[list[float]],
    sample_size: int = 50000,
) -> tuple[list[set[int]], list[set[int]], float]:
    masks = profile_bitmasks(profile)
    similarities: list[float] = []
    step = max(1, len(data.edges) // max(1, sample_size))
    for pos, (ia, ib) in enumerate(data.edges):
        if pos % step != 0:
            continue
        similarities.append(jaccard_masks(masks[ia], masks[ib]))
        if len(similarities) >= sample_size:
            break
    threshold = median(similarities)
    positive_similarities = [value for value in similarities if value > 0.0]
    if threshold <= 0.0 and positive_similarities:
        threshold = median(positive_similarities)

    similar = [set() for _ in data.nodes]
    dissimilar = [set() for _ in data.nodes]
    for ia, ib in data.edges:
        sim = jaccard_masks(masks[ia], masks[ib])
        target = similar if sim >= threshold and sim > 0.0 else dissimilar
        target[ia].add(ib)
        target[ib].add(ia)
    return similar, dissimilar, threshold


def profile_bitmasks(profile: list[list[float]]) -> list[int]:
    masks = []
    for row in profile:
        mask = 0
        for col, value in enumerate(row):
            if value > 0.0:
                mask |= 1 << col
        masks.append(mask)
    return masks


def jaccard_masks(a: int, b: int) -> float:
    union_mask = a | b
    if union_mask == 0:
        return 1.0
    return (a & b).bit_count() / union_mask.bit_count()


def mean_neighbor_features(
    matrix: list[list[float]],
    adjacency: list[set[int]],
) -> list[list[float]]:
    if np is not None and matrix:
        return mean_neighbor_features_numpy(matrix, adjacency)

    width = len(matrix[0]) if matrix else 0
    output = []
    for neighbors in adjacency:
        if not neighbors:
            output.append([0.0] * width)
            continue
        row = [0.0] * width
        for nbr in neighbors:
            nbr_vec = matrix[nbr]
            for col, value in enumerate(nbr_vec):
                row[col] += value
        denom = float(len(neighbors))
        output.append([value / denom for value in row])
    return output


def mean_neighbor_features_numpy(
    matrix: list[list[float]],
    adjacency: list[set[int]],
) -> list[list[float]]:
    arr = np.asarray(matrix, dtype=np.float64)
    rows = arr.shape[0]
    width = arr.shape[1]
    counts = np.fromiter((len(neighbors) for neighbors in adjacency), dtype=np.float64)
    total = int(counts.sum())
    if total == 0:
        return np.zeros((rows, width), dtype=np.float64).tolist()

    src = np.empty(total, dtype=np.int64)
    dst = np.empty(total, dtype=np.int64)
    pos = 0
    for idx, neighbors in enumerate(adjacency):
        size = len(neighbors)
        if size == 0:
            continue
        end = pos + size
        src[pos:end] = idx
        dst[pos:end] = list(neighbors)
        pos = end

    output = np.zeros((rows, width), dtype=np.float64)
    for col in range(width):
        np.add.at(output[:, col], src, arr[dst, col])
    safe_counts = counts.copy()
    safe_counts[safe_counts == 0.0] = 1.0
    output /= safe_counts[:, None]
    return output.tolist()


def gcn_normalized_features(
    matrix: list[list[float]],
    adjacency: list[set[int]],
) -> list[list[float]]:
    if np is not None and matrix:
        return gcn_normalized_features_numpy(matrix, adjacency)

    width = len(matrix[0]) if matrix else 0
    degree = [len(neighbors) + 1 for neighbors in adjacency]
    output = []
    for idx, neighbors in enumerate(adjacency):
        row = [value / degree[idx] for value in matrix[idx]]
        self_scale = 1.0 / math.sqrt(degree[idx])
        for nbr in neighbors:
            scale = self_scale / math.sqrt(degree[nbr])
            nbr_vec = matrix[nbr]
            for col, value in enumerate(nbr_vec):
                row[col] += scale * value
        output.append(row[:width])
    return output


def gcn_normalized_features_numpy(
    matrix: list[list[float]],
    adjacency: list[set[int]],
) -> list[list[float]]:
    arr = np.asarray(matrix, dtype=np.float64)
    rows = arr.shape[0]
    width = arr.shape[1]
    degree = np.fromiter((len(neighbors) + 1 for neighbors in adjacency), dtype=np.float64)
    output = arr / degree[:, None]
    total = int(sum(len(neighbors) for neighbors in adjacency))
    if total == 0:
        return output.tolist()

    src = np.empty(total, dtype=np.int64)
    dst = np.empty(total, dtype=np.int64)
    pos = 0
    for idx, neighbors in enumerate(adjacency):
        size = len(neighbors)
        if size == 0:
            continue
        end = pos + size
        src[pos:end] = idx
        dst[pos:end] = list(neighbors)
        pos = end

    scale = 1.0 / np.sqrt(degree[src] * degree[dst])
    for col in range(width):
        np.add.at(output[:, col], src, arr[dst, col] * scale)
    return output.tolist()


def structural_features(
    data: EgoNetwork,
    similar_adj: list[set[int]],
    dissimilar_adj: list[set[int]],
) -> list[list[float]]:
    degrees = [len(neighbors) for neighbors in data.adjacency]
    max_degree = max(degrees) if degrees else 1
    rows: list[list[float]] = []
    for idx, neighbors in enumerate(data.adjacency):
        degree = degrees[idx]
        if degree:
            avg_neighbor_degree = sum(degrees[nbr] for nbr in neighbors) / degree
        else:
            avg_neighbor_degree = 0.0
        rows.append(
            [
                math.log1p(degree),
                degree / max_degree,
                clustering_coefficient(data.adjacency, idx),
                math.log1p(avg_neighbor_degree),
                len(similar_adj[idx]) / max(1, degree),
                len(dissimilar_adj[idx]) / max(1, degree),
                1.0 if data.is_ego[idx] else 0.0,
            ]
        )
    return rows


def clustering_coefficient(
    adjacency: list[set[int]],
    idx: int,
    max_pairs: int = 5000,
) -> float:
    neighbors = list(adjacency[idx])
    degree = len(neighbors)
    if degree < 2:
        return 0.0
    links = 0
    possible = degree * (degree - 1) / 2
    if possible > max_pairs:
        sampled = 0
        for step in range(max_pairs):
            a_pos = (step * 1103515245 + idx * 12345) % degree
            b_pos = (step * 2654435761 + idx * 97531 + 1) % degree
            if a_pos == b_pos:
                b_pos = (b_pos + 1) % degree
            if neighbors[b_pos] in adjacency[neighbors[a_pos]]:
                links += 1
            sampled += 1
        return links / sampled if sampled else 0.0

    for pos, a in enumerate(neighbors):
        a_neighbors = adjacency[a]
        for b in neighbors[pos + 1 :]:
            if b in a_neighbors:
                links += 1
    return links / possible


def hstack(matrices: list[list[list[float]]]) -> list[list[float]]:
    if not matrices:
        return []
    rows = len(matrices[0])
    output: list[list[float]] = []
    for row_idx in range(rows):
        row: list[float] = []
        for matrix in matrices:
            row.extend(matrix[row_idx])
        output.append(row)
    return output


def zscore(matrix: list[list[float]], train_idx: list[int]) -> list[list[float]]:
    if not matrix:
        return []
    width = len(matrix[0])
    rows = train_idx or list(range(len(matrix)))
    means = []
    scales = []
    for col in range(width):
        values = [matrix[row][col] for row in rows]
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        scale = math.sqrt(var)
        means.append(mean)
        scales.append(scale if scale > 1e-8 else 1.0)
    return [
        [(value - means[col]) / scales[col] for col, value in enumerate(row)]
        for row in matrix
    ]


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0
