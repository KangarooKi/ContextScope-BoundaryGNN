from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import download_snap_gplus
from .experiment import ExperimentConfig, run_experiments


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run ContextScope boundary-aware GNN experiments on SNAP Google+ circles."
    )
    parser.add_argument("--data-dir", default="data/gplus", help="SNAP Google+ data directory")
    parser.add_argument("--download", action="store_true", help="download SNAP gplus.tar.gz")
    parser.add_argument("--ego", type=int, action="append", help="ego id to run; repeatable")
    parser.add_argument("--max-egos", type=int, help="run only the first N ego networks")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-profile-dims", type=int, default=64)
    parser.add_argument("--logistic-epochs", type=int, default=180)
    parser.add_argument(
        "--include-ablations",
        action="store_true",
        help="add BoundaryGNN ablation variants on the Torch backend",
    )
    parser.add_argument(
        "--include-mlp",
        action="store_true",
        help="add larger BoundaryGNN MLP heads on the Torch backend",
    )
    parser.add_argument(
        "--include-mlp-ablations",
        action="store_true",
        help="add high-capacity DeepMLP ablations on the Torch backend",
    )
    parser.add_argument(
        "--include-gated-mlp",
        action="store_true",
        help="add channel-gated high-capacity BoundaryGNN on the Torch backend",
    )
    parser.add_argument(
        "--include-node-gated-mlp",
        action="store_true",
        help="add node-adaptive channel-gated high-capacity BoundaryGNN on the Torch backend",
    )
    parser.add_argument(
        "--include-soft-labels",
        action="store_true",
        help="add soft-label BoundaryGNN variants on the Torch backend",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="cpu",
        help="execution backend; cuda uses the Torch GPU experiment path",
    )
    parser.add_argument("--output", default="outputs/report.json")
    args = parser.parse_args(argv)
    if args.device == "cpu" and (
        args.include_ablations
        or args.include_mlp
        or args.include_mlp_ablations
        or args.include_gated_mlp
        or args.include_node_gated_mlp
        or args.include_soft_labels
    ):
        parser.error(
            "--include-ablations, --include-mlp, --include-mlp-ablations, "
            "--include-gated-mlp, --include-node-gated-mlp, and --include-soft-labels require "
            "--device cuda or --device auto"
        )

    data_dir = Path(args.data_dir)
    if args.download:
        print(f"Downloading SNAP Google+ circles into {data_dir} ...")
        data_dir = download_snap_gplus(data_dir)
        print(f"Using extracted data at {data_dir}")

    config = ExperimentConfig(
        data_dir=data_dir,
        ego_ids=args.ego,
        seed=args.seed,
        max_profile_dims=args.max_profile_dims,
        logistic_epochs=args.logistic_epochs,
        max_egos=args.max_egos,
        include_ablations=args.include_ablations,
        include_mlp=args.include_mlp,
        include_mlp_ablations=args.include_mlp_ablations,
        include_gated_mlp=args.include_gated_mlp,
        include_node_gated_mlp=args.include_node_gated_mlp,
        include_soft_labels=args.include_soft_labels,
    )
    if args.device == "cpu":
        report = run_experiments(config)
    else:
        from .gpu_experiment import run_gpu_experiments

        report = run_gpu_experiments(config, args.device)
    print_report(report)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved JSON report to {output}")
    return 0


def print_report(report: dict[str, object]) -> None:
    aggregate = report["aggregate"]
    assert isinstance(aggregate, dict)
    method_width = max([20, *(len(str(name)) for name in aggregate)])
    print("\nAggregate test metrics")
    print("-" * (method_width + 62))
    print(
        f"{'method':<{method_width}} "
        f"{'n':>6} {'acc':>8} {'prec':>8} {'rec':>8} {'f1':>8} {'auc':>8}"
    )
    print("-" * (method_width + 62))
    for name, row in aggregate.items():
        assert isinstance(row, dict)
        print(
            f"{name:<{method_width}} "
            f"{row.get('n', 0.0):>6.0f} "
            f"{row.get('accuracy', 0.0):>8.3f} "
            f"{row.get('precision', 0.0):>8.3f} "
            f"{row.get('recall', 0.0):>8.3f} "
            f"{row.get('f1', 0.0):>8.3f} "
            f"{row.get('auc', 0.0):>8.3f}"
        )
    model_info = report.get("model_info", {})
    if isinstance(model_info, dict) and model_info:
        print("\nModel size summary")
        print("-" * (method_width + 33))
        print(f"{'method':<{method_width}} {'feat_dim':>10} {'params':>12}")
        print("-" * (method_width + 33))
        for name, row in model_info.items():
            if not isinstance(row, dict):
                continue
            print(
                f"{name:<{method_width}} "
                f"{row.get('feature_dim_mean', 0.0):>10.1f} "
                f"{row.get('parameter_count_mean', 0.0):>12.1f}"
            )
        gate_rows = [
            (name, row)
            for name, row in model_info.items()
            if isinstance(row, dict) and any(key.startswith("gate_") for key in row)
        ]
        if gate_rows:
            print("\nLearned channel gates")
            print("-" * (method_width + 58))
            print(
                f"{'method':<{method_width}} "
                f"{'self':>8} {'all':>8} {'similar':>8} "
                f"{'dissim':>8} {'struct':>8}"
            )
            print("-" * (method_width + 58))
            for name, row in gate_rows:
                print(
                    f"{name:<{method_width}} "
                    f"{row.get('gate_self_mean', 0.0):>8.3f} "
                    f"{row.get('gate_all_mean', 0.0):>8.3f} "
                    f"{row.get('gate_similar_mean', 0.0):>8.3f} "
                    f"{row.get('gate_dissimilar_mean', 0.0):>8.3f} "
                    f"{row.get('gate_structure_mean', 0.0):>8.3f}"
                )
    per_ego = report["per_ego"]
    assert isinstance(per_ego, list)
    print(f"\nCompleted {len(per_ego)} ego-network experiment(s).")
    interventions = report.get("interventions", [])
    if isinstance(interventions, list) and interventions:
        first = interventions[0]
        assert isinstance(first, dict)
        print(
            f"Example intervention list: ego={first.get('ego_id')}, "
            f"circle={first.get('target_circle')}"
        )
        suggestions = first.get("suggestions", [])
        if isinstance(suggestions, list):
            for item in suggestions[:3]:
                assert isinstance(item, dict)
                print(
                    "  "
                    f"node={item['node_id']} "
                    f"risk={item['predicted_risk']:.3f} "
                    f"outside_edges={item['outside_circle_edges']:.0f} "
                    f"action={item['action']}"
                )
