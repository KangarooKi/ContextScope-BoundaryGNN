from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from contextscope.data import derive_boundary_labels, load_ego_network
from contextscope.experiment import ExperimentConfig, run_experiments


class PipelineTest(unittest.TestCase):
    def test_load_labels_and_run_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_tiny_snap(root)
            data = load_ego_network(root, 0)
            self.assertEqual(data.ego_id, 0)
            self.assertEqual(len(data.circles), 3)
            labels, eligible, _ = derive_boundary_labels(data)
            self.assertGreaterEqual(sum(eligible), 8)
            self.assertGreaterEqual(sum(labels), 2)

            report = run_experiments(
                ExperimentConfig(
                    data_dir=root,
                    ego_ids=[0],
                    seed=3,
                    max_profile_dims=4,
                    logistic_epochs=8,
                    min_eligible_nodes=8,
                )
            )
            self.assertEqual(
                set(report["aggregate"]),
                {
                    "ProfileLogit",
                    "StructureLogit",
                    "ProfileStructureLogit",
                    "BoundaryGNN-Logit",
                },
            )
            self.assertGreater(report["aggregate"]["BoundaryGNN-Logit"]["n"], 0)


def write_tiny_snap(root: Path) -> None:
    (root / "0.feat").write_text(
        "\n".join(
            [
                "1 1 0 0 0",
                "2 1 1 0 0",
                "3 1 0 1 0",
                "4 1 1 1 0",
                "5 0 1 1 0",
                "6 0 1 0 1",
                "7 0 0 1 1",
                "8 0 0 0 1",
            ]
        ),
        encoding="utf-8",
    )
    (root / "0.egofeat").write_text("1 0 0 1\n", encoding="utf-8")
    (root / "0.edges").write_text(
        "\n".join(
            [
                "1 2",
                "1 3",
                "2 3",
                "3 4",
                "4 5",
                "5 6",
                "6 7",
                "7 8",
                "4 7",
                "2 6",
            ]
        ),
        encoding="utf-8",
    )
    (root / "0.circles").write_text(
        "\n".join(
            [
                "circleA 1 2 3 4",
                "circleB 3 4 5 6",
                "circleC 6 7 8",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
