# Core Code Map

This document summarizes the GitHub-facing core code for ContextScope.

## Entry Points

| File | Role |
|---|---|
| `contextscope/__main__.py` | Enables `python -m contextscope`. |
| `contextscope/cli.py` | Command-line arguments, report writing, and backend selection. |
| `scripts/summarize_multiseed.py` | Aggregates multiple JSON reports into mean/std tables. |

Typical commands:

```bash
python -m contextscope --download --data-dir data/gplus --output outputs/report.json

python -m contextscope \
  --data-dir data/gplus \
  --device cuda \
  --include-gated-mlp \
  --include-node-gated-mlp \
  --output outputs/report_node_gated.json

python scripts/summarize_multiseed.py \
  outputs/node_multiseed/report_seed_*_node_gated.json \
  --output-md outputs/node_multiseed/summary_node_gated.md
```

## Data and Weak Labels

| File / Function | Role |
|---|---|
| `contextscope/data.py::load_ego_network` | Loads SNAP ego-network edges, profile features, ego features, and circle annotations. |
| `contextscope/data.py::derive_boundary_labels` | Constructs weak boundary-risk labels from multi-circle membership and outside-circle exposure. |
| `contextscope/data.py::stratified_split` | Creates train / validation / test splits while preserving class balance. |

Label logic:

```text
boundary_risk(v) = 1
if v belongs to multiple circles
or v connects to multiple outside circles with high outside-neighbor ratio
```

The validation set is used only for threshold selection; it is not used for training.

## BoundaryGNN Feature Construction

| File / Function | Role |
|---|---|
| `contextscope/features.py::build_feature_views` | CPU feature construction path. |
| `contextscope/gpu_experiment.py::build_torch_feature_views` | CUDA/Torch feature construction path. |
| `contextscope/gpu_experiment.py::edge_similarity_mask` | Splits edges into profile-similar and profile-dissimilar relations using Jaccard similarity. |
| `contextscope/gpu_experiment.py::aggregate_edges` | Computes mean neighbor features for all/similar/dissimilar edge sets. |
| `contextscope/gpu_experiment.py::build_structure_tensor` | Builds structural features such as degree, clustering coefficient, and similar/dissimilar ratios. |

BoundaryGNN representation:

```text
h_v = concat(
  self profile,
  all-neighbor mean,
  similar-neighbor mean,
  dissimilar-neighbor mean,
  structural features
)
```

## Models

| File / Class or Function | Role |
|---|---|
| `contextscope/models.py::LogisticRegressionGD` | Lightweight CPU logistic baseline. |
| `contextscope/gpu_experiment.py::train_torch_logit` | Torch logistic baselines and BoundaryGNN-Logit. |
| `contextscope/gpu_experiment.py::train_torch_mlp` | MLP / DeepMLP prediction heads. |
| `contextscope/gpu_experiment.py::ChannelGatedMLP` | Global channel gate: one shared gate vector for all nodes. |
| `contextscope/gpu_experiment.py::NodeGatedMLP` | Node-adaptive channel gate: each node gets its own channel weights. |
| `contextscope/gpu_experiment.py::train_torch_node_gated_mlp` | Training loop for the final NodeGated model. |

Final NodeGated formula:

```text
alpha_v = GateMLP([x_v || a_v || m_v || d_v || s_v])

h_v = [
  alpha_v,self x_v ||
  alpha_v,all a_v ||
  alpha_v,similar m_v ||
  alpha_v,dissimilar d_v ||
  alpha_v,structure s_v
]
```

## Experiment Orchestration

| File / Function | Role |
|---|---|
| `contextscope/experiment.py::run_experiments` | CPU experiment runner. |
| `contextscope/gpu_experiment.py::run_gpu_experiments` | CUDA experiment runner over all ego networks. |
| `contextscope/gpu_experiment.py::run_one_ego_gpu` | Runs all selected methods on one ego network. |
| `contextscope/evaluation.py::evaluate_binary` | Computes Accuracy, Precision, Recall, F1, and AUC. |
| `contextscope/evaluation.py::aggregate_metric_rows` | Aggregates per-ego metrics into dataset-level metrics. |

## GitHub Upload Notes

Files and folders intended for GitHub:

```text
contextscope/
docs/
scripts/
tests/
README.md
pyproject.toml
requirements.txt
.gitignore
```

Files and folders that should stay local:

```text
data/
outputs/
__pycache__/
*.pyc
.venv/
```

The dataset should be downloaded by users with:

```bash
python -m contextscope --download --data-dir data/gplus --output outputs/report.json
```
