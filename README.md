# ContextScope: Node-Adaptive BoundaryGNN for Social Boundary-Risk Detection

**English** | [中文](README.zh-CN.md)

ContextScope builds a new graph learning task on the public SNAP Google+ Circles dataset:
**social boundary-risk node detection**. Instead of community discovery or friend recommendation, the task asks whether a user is positioned on the boundary of multiple social circles and may expose information from one circle to another.

The project starts from **BoundaryGNN-Logit**, a lightweight and interpretable boundary-aware graph representation method. It splits social edges by profile similarity, aggregates different neighbor contexts, and predicts whether a node is boundary-risk. The final model extends this idea with **NodeGated BoundaryGNN**, where each node learns its own channel weights over self, all-neighbor, similar-neighbor, dissimilar-neighbor, and structural information.

![BoundaryGNN prediction flow](docs/figures/presentation/boundarygnn_prediction_flow.png)

## Highlights

- **New task**: reconstruct Google+ Circles from social-circle discovery into boundary-risk node detection.
- **Weak supervision**: derive binary labels from public circle annotations without manual relabeling.
- **Boundary-aware aggregation**: split edges by profile Jaccard similarity and aggregate similar and dissimilar neighbors separately.
- **Node-level channel gate**: learn node-specific channel weights to reduce redundancy caused by fixed feature concatenation.
- **Interpretability**: connect predictions to multi-circle membership, cross-circle exposure, neighbor heterogeneity, and structural position.
- **Complete experimental path**: lightweight baselines, capacity expansion, ablation study, global gate, node-adaptive gate, and 5-seed stability tests.

## Visual Overview

**Node-Adaptive Gate**

![Node-adaptive gate](docs/figures/presentation/node_adaptive_gate.png)

**Final 5-Seed Result**

![Final multi-seed result](docs/figures/presentation/final_multiseed_result.png)

More presentation-ready figures are available in `docs/figures/presentation/`, including task reconstruction, label construction, main results, capacity expansion, ablation study, and global channel gate diagrams.

## Dataset

The project uses the public SNAP **Social circles: Google+** dataset:

- Dataset page: <https://snap.stanford.edu/data/ego-Gplus.html>
- Original paper: J. McAuley and J. Leskovec, *Learning to Discover Social Circles in Ego Networks*, NIPS 2012.
- Raw files: ego-network edges, anonymized binary profile features, circle annotations, ego-user features, and follower information.

This project does not use the original circle prediction task directly. It reconstructs a weakly supervised benchmark for boundary-risk node detection.

### Reconstructed Data Scale

| Item | Count |
|---|---:|
| Raw Google+ ego networks | 132 |
| Valid ego networks used | 122 |
| Raw archive size | 393 MB |
| Extracted data size | 2.4 GB |
| Ego-graph node instances | 254,556 |
| Ego-graph edge instances | 26,954,330 |
| Labeled nodes | 39,204 |
| Positive samples | 22,261 |
| Negative samples | 16,943 |
| Train split | 23,523 |
| Validation split | 7,847 |
| Test split | 7,834 |

Ten ego networks are filtered out because they contain too few valid labeled samples for a stable train, validation, and test split. The final benchmark uses 122 ego networks.

## Task Definition

For each ego network:

```text
G = (V, E), X, C
```

where:

- `V` is the set of user nodes;
- `E` is the social edge set;
- `X` is the anonymized profile feature matrix;
- `C` is the Google+ circle annotation set.

The goal is to predict a binary label for each labeled node:

```text
boundary_risk(v) in {0, 1}
```

A node is more likely to be boundary-risk if it belongs to multiple circles or connects strongly to circles outside its own circle context.

## Label Construction

The original dataset does not contain explicit boundary-risk labels. We derive weak labels from circle annotations. A node is marked as positive mainly when:

```text
1. It has multi-circle membership; or
2. It connects to multiple outside circles and has a high outside-neighbor ratio.
```

Implementation:

- `contextscope/data.py`
- Function: `derive_boundary_labels`

In the final CUDA experiments, the validation set is only used for threshold selection. It is not mixed into training and is not merged with the test set.

## Method: BoundaryGNN-Logit

BoundaryGNN-Logit is a lightweight and interpretable boundary-aware graph representation method:

```text
Google+ Ego Network
-> construct boundary-risk labels
-> compute profile similarity on edges
-> split similar and dissimilar edges
-> aggregate different neighbor channels
-> concatenate node representation
-> predict boundary-risk probability
```

### Profile Similarity

For each social edge `(u, v)`, the Jaccard similarity between two profile feature sets is:

```text
sim(u, v) = |P_u intersection P_v| / |P_u union P_v|
```

Edges are then split by the similarity threshold:

```text
high sim(u, v) -> Similar Edge
low sim(u, v)  -> Dissimilar Edge
```

### Boundary-Aware Aggregation

For each node `v`, BoundaryGNN builds five feature channels:

```text
Self Profile
All-Neighbor Mean
Similar-Neighbor Mean
Dissimilar-Neighbor Mean
Structural Features
```

The final representation is:

```text
h_v = concat(x_v, all_v, similar_v, dissimilar_v, s_v)
```

where:

- `x_v`: profile features of the target node;
- `all_v`: mean profile of all neighbors;
- `similar_v`: mean profile of profile-similar neighbors;
- `dissimilar_v`: mean profile of profile-dissimilar neighbors;
- `s_v`: structural features such as degree, clustering coefficient, and similar or dissimilar edge ratios.

The lightweight classifier is:

```text
p_v = sigmoid(W h_v + b)
```

## Final Model: NodeGated BoundaryGNN

Fixed concatenation implicitly assumes that all feature channels are equally useful for all nodes. The ablation study shows that this is not always true: the similar-neighbor channel can be useful for some nodes but redundant or noisy for others.

NodeGated BoundaryGNN learns a node-specific gate vector:

```text
alpha_v = GateMLP(concat(self_v, all_v, similar_v, dissimilar_v, structure_v))
```

and applies it before the final DeepMLP classifier:

```text
h_v = concat(
  alpha_v,self * self_v,
  alpha_v,all * all_neighbor_v,
  alpha_v,similar * similar_neighbor_v,
  alpha_v,dissimilar * dissimilar_neighbor_v,
  alpha_v,structure * structure_v
)
```

The gate is initialized around 1, so the model starts close to the full BoundaryGNN representation and then learns node-level channel selection during training.

## Baselines

The default comparison uses three lightweight baselines plus the proposed method:

| Method | Description |
|---|---|
| `ProfileLogit` | Logistic baseline using profile features only |
| `StructureLogit` | Logistic baseline using graph structural features only |
| `ProfileStructureLogit` | Profile plus structure baseline without neighbor aggregation |
| `BoundaryGNN-Logit` | Boundary-aware aggregation with similar and dissimilar edge typing |

## Main Result

The following results are from the full Google+ Circles benchmark on CUDA.

| Method | Accuracy | Precision | Recall | F1 | AUC |
|---|---:|---:|---:|---:|---:|
| BoundaryGNN-Logit | 0.7821 | 0.7112 | 0.8202 | 0.7379 | 0.7592 |
| StructureLogit | 0.7404 | 0.6698 | 0.8359 | 0.7140 | 0.7199 |
| ProfileStructureLogit | 0.7421 | 0.6587 | 0.8133 | 0.7041 | 0.6704 |
| ProfileLogit | 0.6529 | 0.5986 | 0.8983 | 0.6828 | 0.5301 |

Result files:

```text
outputs/report_gplus_full_main_cuda.json
outputs/report_gplus_full_main_cuda.log
```

## Extended Experiments

### Capacity Expansion

Using larger MLP prediction heads improves performance, but does not explain the whole gain. The best high-capacity head reaches:

| Method | F1 | AUC | Avg Params |
|---|---:|---:|---:|
| `BoundaryGNN-DeepMLP-2048x1024` | 0.7554 | 0.7954 | 2,612,712 |
| `BoundaryGNN-Logit` | 0.7367 | 0.7600 | 248 |

Full capacity results:

| Method | Accuracy | Precision | Recall | F1 | AUC | Avg Params |
|---|---:|---:|---:|---:|---:|---:|
| `BoundaryGNN-DeepMLP-2048x1024` | 0.8152 | 0.7383 | 0.8058 | 0.7554 | 0.7954 | 2,612,712 |
| `BoundaryGNN-DeepMLP-1024x512` | 0.8102 | 0.7389 | 0.8078 | 0.7542 | 0.7895 | 782,068 |
| `BoundaryGNN-MLP-512` | 0.8080 | 0.7344 | 0.8105 | 0.7524 | 0.7902 | 128,379 |
| `BoundaryGNN-MLP-128` | 0.7984 | 0.7338 | 0.8127 | 0.7487 | 0.7811 | 32,095 |
| `BoundaryGNN-Logit` | 0.7799 | 0.7041 | 0.8192 | 0.7367 | 0.7600 | 248 |

### Large-Head Ablation

The strongest DeepMLP head is used for all ablation variants.

| Method | Accuracy | Precision | Recall | F1 | AUC | Avg Params |
|---|---:|---:|---:|---:|---:|---:|
| `BoundaryGNN-DeepMLP-2048x1024` | 0.8153 | 0.7384 | 0.8058 | 0.7554 | 0.7953 | 2,612,712 |
| `BoundaryGNN-DeepMLP-2048x1024-w/o-All` | 0.7863 | 0.7169 | 0.8270 | 0.7449 | 0.7811 | 2,489,966 |
| `BoundaryGNN-DeepMLP-2048x1024-w/o-Similar` | 0.8110 | 0.7333 | 0.8247 | 0.7581 | 0.7951 | 2,489,966 |
| `BoundaryGNN-DeepMLP-2048x1024-w/o-Dissimilar` | 0.8027 | 0.7317 | 0.8184 | 0.7523 | 0.7955 | 2,489,966 |
| `BoundaryGNN-DeepMLP-2048x1024-w/o-Structure` | 0.7998 | 0.7208 | 0.8121 | 0.7461 | 0.7747 | 2,598,376 |
| `BoundaryGNN-DeepMLP-2048x1024-w/o-EdgeTyping` | 0.8078 | 0.7354 | 0.8213 | 0.7548 | 0.7962 | 2,367,220 |

Key observations:

- Removing `All-Neighbor Mean` reduces F1 from 0.7554 to 0.7449.
- Removing `Structural Features` reduces AUC from 0.7953 to 0.7747.
- Removing `Similar-Neighbor Mean` slightly increases F1, suggesting that this channel can be redundant under a high-capacity prediction head.
- This motivates learnable channel selection instead of fixed concatenation.

### Global Channel Gate

The global gate learns one shared scalar weight for each channel:

```text
h_v = concat(
  alpha_self * self,
  alpha_all * all_neighbor,
  alpha_similar * similar_neighbor,
  alpha_dissimilar * dissimilar_neighbor,
  alpha_structure * structure
)
```

It improves F1 but still gives the same channel weights to all nodes.

| Method | Accuracy | Precision | Recall | F1 | AUC | Avg Params |
|---|---:|---:|---:|---:|---:|---:|
| `BoundaryGNN-GatedDeepMLP-2048x1024` | 0.8171 | 0.7472 | 0.8071 | 0.7588 | 0.7942 | 2,612,717 |
| `BoundaryGNN-DeepMLP-2048x1024` | 0.8153 | 0.7384 | 0.8058 | 0.7554 | 0.7954 | 2,612,712 |
| `BoundaryGNN-Logit` | 0.7799 | 0.7041 | 0.8192 | 0.7367 | 0.7600 | 248 |

Average global gate weights:

| Channel | Gate |
|---|---:|
| `self` | 1.002 |
| `all` | 1.003 |
| `similar` | 0.997 |
| `dissimilar` | 0.998 |
| `structure` | 0.998 |

### Node-Adaptive Gate

The node-level gate learns a different channel vector for each node. It improves risk ranking and becomes the final model.

Single-seed result:

| Method | Accuracy | Precision | Recall | F1 | AUC | Avg Params |
|---|---:|---:|---:|---:|---:|---:|
| `BoundaryGNN-NodeGatedDeepMLP-2048x1024` | 0.8108 | 0.7463 | 0.8214 | 0.7595 | 0.7992 | 2,645,323 |
| `BoundaryGNN-GatedDeepMLP-2048x1024` | 0.8171 | 0.7473 | 0.8087 | 0.7595 | 0.7943 | 2,612,717 |
| `BoundaryGNN-DeepMLP-2048x1024` | 0.8153 | 0.7384 | 0.8058 | 0.7554 | 0.7954 | 2,612,712 |
| `BoundaryGNN-Logit` | 0.7803 | 0.7041 | 0.8196 | 0.7369 | 0.7600 | 248 |

Average single-seed gate weights:

| Method | self | all | similar | dissimilar | structure |
|---|---:|---:|---:|---:|---:|
| `BoundaryGNN-GatedDeepMLP-2048x1024` | 1.002 | 1.003 | 0.997 | 0.998 | 0.998 |
| `BoundaryGNN-NodeGatedDeepMLP-2048x1024` | 0.952 | 1.017 | 0.898 | 0.912 | 0.923 |

### Final 5-Seed Result

The final comparison is averaged over five random seeds: `1, 3, 5, 7, 9`.

| Method | Runs | F1 mean | F1 std | AUC mean | AUC std | Accuracy mean | Params |
|---|---:|---:|---:|---:|---:|---:|---:|
| `BoundaryGNN-NodeGatedDeepMLP-2048x1024` | 5 | 0.7618 | 0.0035 | 0.7955 | 0.0035 | 0.8061 | 2,645,323 |
| `BoundaryGNN-GatedDeepMLP-2048x1024` | 5 | 0.7597 | 0.0007 | 0.7923 | 0.0053 | 0.8087 | 2,612,717 |
| `BoundaryGNN-DeepMLP-2048x1024` | 5 | 0.7568 | 0.0030 | 0.7921 | 0.0042 | 0.8074 | 2,612,712 |
| `BoundaryGNN-Logit` | 5 | 0.7363 | 0.0060 | 0.7579 | 0.0060 | 0.7756 | 248 |
| `StructureLogit` | 5 | 0.7184 | 0.0066 | 0.7208 | 0.0048 | 0.7517 | 8 |
| `ProfileStructureLogit` | 5 | 0.7046 | 0.0036 | 0.6699 | 0.0083 | 0.7298 | 68 |
| `ProfileLogit` | 5 | 0.6780 | 0.0032 | 0.5387 | 0.0239 | 0.6556 | 61 |

Five-seed average gate weights:

| Method | self | all | similar | dissimilar | structure |
|---|---:|---:|---:|---:|---:|
| `BoundaryGNN-GatedDeepMLP-2048x1024` | 1.002 | 1.003 | 0.997 | 0.998 | 0.998 |
| `BoundaryGNN-NodeGatedDeepMLP-2048x1024` | 0.944 | 1.017 | 0.895 | 0.919 | 0.937 |

### Soft-Label Exploration

A soft-label variant was also tested. It uses normalized bridge scores as continuous targets while evaluating against the original hard labels. It improves recall for some variants but reduces precision, F1, and AUC. Therefore, the final method keeps hard-label training and reports soft-label training as a negative exploration result.

## Usage

### 1. Prepare Data

Place the SNAP Google+ Circles archive at:

```text
data/gplus/gplus.tar.gz
```

After extraction, the folder should look like:

```text
data/gplus/gplus/
├── *.edges
├── *.feat
├── *.egofeat
├── *.featnames
├── *.circles
└── *.followers
```

The program can also download the dataset:

```bash
python3 -m contextscope --download --data-dir data/gplus --output outputs/report.json
```

### 2. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

Or install the project in editable mode:

```bash
python -m pip install -e ".[cuda]"
```

### 3. Quick Smoke Test

```bash
python3 -m contextscope \
  --data-dir data/gplus \
  --max-egos 10 \
  --output outputs/report_gplus_10.json
```

### 4. Full CUDA Experiment

```bash
python3 -m contextscope \
  --data-dir data/gplus \
  --logistic-epochs 60 \
  --device cuda \
  --output outputs/report_gplus_full_main_cuda.json
```

### 5. Extended Experiments

Run ablations and MLP capacity variants:

```bash
python3 -m contextscope \
  --data-dir data/gplus \
  --logistic-epochs 60 \
  --device cuda \
  --include-ablations \
  --include-mlp \
  --output outputs/report_gplus_full_extended_cuda.json
```

Run node-gated variants:

```bash
python3 -m contextscope \
  --data-dir data/gplus \
  --seed 7 \
  --logistic-epochs 80 \
  --device cuda \
  --include-gated-mlp \
  --include-node-gated-mlp \
  --output outputs/report_gplus_full_node_gated_cuda.json
```

### 6. Tests

```bash
python3 -m unittest discover
```

## Requirements

The CPU path mainly depends on the Python standard library. NumPy accelerates some feature aggregation routines when available. CUDA training requires PyTorch.

Recommended environment:

```text
Python >= 3.10
NumPy optional
PyTorch required for --device cuda
```

