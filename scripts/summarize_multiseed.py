from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev


METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "auc"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize ContextScope reports across multiple random seeds."
    )
    parser.add_argument("reports", nargs="+", help="report JSON files")
    parser.add_argument("--output-json", help="optional path for summary JSON")
    parser.add_argument("--output-md", help="optional path for Markdown table")
    args = parser.parse_args()

    reports = [load_report(Path(path)) for path in args.reports]
    summary = summarize_reports(reports)
    markdown = to_markdown(summary)
    print(markdown)

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.output_md:
        Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_md).write_text(markdown + "\n", encoding="utf-8")
    return 0


def load_report(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    report["_path"] = str(path)
    return report


def summarize_reports(reports: list[dict[str, object]]) -> dict[str, object]:
    by_method: dict[str, list[dict[str, float]]] = {}
    model_info: dict[str, list[dict[str, float]]] = {}
    seeds = []
    completed = []
    for report in reports:
        config = report.get("config", {})
        if isinstance(config, dict):
            seeds.append(config.get("seed"))
        per_ego = report.get("per_ego", [])
        if isinstance(per_ego, list):
            completed.append(len(per_ego))
        aggregate = report.get("aggregate", {})
        if isinstance(aggregate, dict):
            for method, row in aggregate.items():
                if isinstance(row, dict):
                    by_method.setdefault(str(method), []).append(row)
        info = report.get("model_info", {})
        if isinstance(info, dict):
            for method, row in info.items():
                if isinstance(row, dict):
                    model_info.setdefault(str(method), []).append(row)

    methods = {}
    for method, rows in sorted(by_method.items()):
        method_summary: dict[str, float] = {"runs": float(len(rows))}
        n_values = [float(row.get("n", 0.0)) for row in rows]
        method_summary["n_mean"] = mean(n_values) if n_values else 0.0
        for key in METRIC_KEYS:
            values = [float(row[key]) for row in rows if key in row]
            method_summary[f"{key}_mean"] = mean(values) if values else 0.0
            method_summary[f"{key}_std"] = stdev(values) if len(values) >= 2 else 0.0
        if method in model_info:
            params = [
                float(row.get("parameter_count_mean", 0.0))
                for row in model_info[method]
                if "parameter_count_mean" in row
            ]
            if params:
                method_summary["parameter_count_mean"] = mean(params)
            for gate_name in ["self", "all", "similar", "dissimilar", "structure"]:
                key = f"gate_{gate_name}_mean"
                gates = [float(row[key]) for row in model_info[method] if key in row]
                if gates:
                    method_summary[f"{key}_across_seeds"] = mean(gates)
                    method_summary[f"gate_{gate_name}_std_across_seeds"] = (
                        stdev(gates) if len(gates) >= 2 else 0.0
                    )
        methods[method] = method_summary

    return {
        "seeds": seeds,
        "completed_ego_networks": completed,
        "num_reports": len(reports),
        "methods": methods,
    }


def to_markdown(summary: dict[str, object]) -> str:
    methods = summary["methods"]
    assert isinstance(methods, dict)
    lines = [
        "| Method | Runs | F1 mean | F1 std | AUC mean | AUC std | Accuracy mean | Params |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, row in sorted(
        methods.items(),
        key=lambda item: item[1].get("f1_mean", 0.0) if isinstance(item[1], dict) else 0.0,
        reverse=True,
    ):
        assert isinstance(row, dict)
        params = row.get("parameter_count_mean", 0.0)
        lines.append(
            "| {} | {:.0f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.0f} |".format(
                method,
                row.get("runs", 0.0),
                row.get("f1_mean", 0.0),
                row.get("f1_std", 0.0),
                row.get("auc_mean", 0.0),
                row.get("auc_std", 0.0),
                row.get("accuracy_mean", 0.0),
                params,
            )
        )

    gated_rows = [
        (method, row)
        for method, row in methods.items()
        if isinstance(row, dict) and "gate_self_mean_across_seeds" in row
    ]
    if gated_rows:
        lines.extend(
            [
                "",
                "| Method | self gate | all gate | similar gate | dissimilar gate | structure gate |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for method, row in gated_rows:
            lines.append(
                "| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
                    method,
                    row.get("gate_self_mean_across_seeds", 0.0),
                    row.get("gate_all_mean_across_seeds", 0.0),
                    row.get("gate_similar_mean_across_seeds", 0.0),
                    row.get("gate_dissimilar_mean_across_seeds", 0.0),
                    row.get("gate_structure_mean_across_seeds", 0.0),
                )
            )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
