"""Mac-safe tests for the narrow author-instrumentation boundary."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from latent_grpo_runner.upstream_adapter import (
    BufferedObserver,
    NoOpObserver,
    OCPSelectionFacts,
    attach_stable_ids,
    attach_stable_ids_to_batch,
    load_observer_from_env,
    ocp_selection_event,
)


ROOT = Path(__file__).resolve().parents[2]


class UpstreamAdapterTests(unittest.TestCase):
    def test_observer_is_an_explicit_noop_when_disabled(self) -> None:
        with patch.dict(os.environ, {"LATENT_GRPO_OBSERVER_ENABLED": "0"}, clear=False):
            observer = load_observer_from_env()

        self.assertIsInstance(observer, NoOpObserver)
        self.assertFalse(observer.enabled)
        self.assertIsNone(observer.emit("post_repeat", {"full_tensor": object()}))

    def test_enabled_observer_keeps_only_small_plain_events(self) -> None:
        observer = BufferedObserver(max_events=2)
        observer.emit("optimizer_update", {"did_step": True, "update_count": 1})
        observer.emit("eval_question", {"question_id": "q1", "is_correct": False})

        self.assertEqual(
            observer.drain(),
            [
                {"event_type": "optimizer_update", "did_step": True, "update_count": 1},
                {"event_type": "eval_question", "question_id": "q1", "is_correct": False},
            ],
        )

    def test_ids_are_deterministic_and_survive_filter_and_reorder(self) -> None:
        group_ids, trajectory_ids = attach_stable_ids(
            global_step=7,
            prompt_identities=["train:10", "train:10", "train:11", "train:11"],
        )
        reordered = [(group_ids[index], trajectory_ids[index]) for index in (3, 0, 2)]

        self.assertEqual(group_ids[0], group_ids[1])
        self.assertNotEqual(group_ids[0], group_ids[2])
        self.assertEqual(trajectory_ids, [0, 1, 0, 1])
        self.assertEqual(reordered[1], (group_ids[0], 0))
        self.assertEqual(
            attach_stable_ids(7, ["train:10", "train:10", "train:11", "train:11"]),
            (group_ids, trajectory_ids),
        )

    def test_disabled_batch_adapter_leaves_dataproto_like_input_unchanged(self) -> None:
        class SyntheticDataProto:
            def __init__(self) -> None:
                self.non_tensor_batch = {
                    "prompt_identity": ["train:10", "train:10"],
                    "uid": ["upstream-uid", "upstream-uid"],
                }

        batch = SyntheticDataProto()
        before = dict(batch.non_tensor_batch)

        returned = attach_stable_ids_to_batch(batch, global_step=7, observer=NoOpObserver())

        self.assertIs(returned, batch)
        self.assertEqual(batch.non_tensor_batch, before)

    def test_enabled_batch_adapter_attaches_ids_and_emits_plain_event(self) -> None:
        if importlib.util.find_spec("numpy") is None:
            self.skipTest("numpy unavailable; real DataProto array contract is target-machine deferred")
        import numpy as np

        class SyntheticDataProto:
            def __init__(self) -> None:
                self.non_tensor_batch = {
                    "prompt_identity": np.array(["train:10", "train:10", "train:11"], dtype=object),
                }

        batch = SyntheticDataProto()
        observer = BufferedObserver()

        returned = attach_stable_ids_to_batch(batch, global_step=7, observer=observer)

        self.assertIs(returned, batch)
        self.assertIsInstance(batch.non_tensor_batch["group_id"], np.ndarray)
        self.assertIsInstance(batch.non_tensor_batch["trajectory_id"], np.ndarray)
        self.assertEqual(batch.non_tensor_batch["group_id"].dtype, np.dtype(object))
        self.assertTrue(np.issubdtype(batch.non_tensor_batch["trajectory_id"].dtype, np.integer))
        self.assertEqual(batch.non_tensor_batch["trajectory_id"].tolist(), [0, 1, 0])
        self.assertEqual(
            observer.drain(),
            [{"event_type": "post_repeat_ids", "global_step": 7, "trajectory_count": 3, "group_count": 2}],
        )

    def test_enabled_batch_adapter_fails_closed_when_numpy_is_unavailable(self) -> None:
        class SyntheticDataProto:
            non_tensor_batch = {"prompt_identity": ["train:10", "train:10"]}

        with patch.dict(sys.modules, {"numpy": None}):
            with self.assertRaisesRegex(RuntimeError, "NumPy is required"):
                attach_stable_ids_to_batch(SyntheticDataProto(), global_step=7, observer=BufferedObserver())

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy unavailable; DataProto contract deferred")
    def test_numpy_identity_columns_survive_repeat_filter_reorder_and_balance_indices(self) -> None:
        import numpy as np

        class MinimalDataProto:
            def __init__(self, non_tensor_batch):
                self.non_tensor_batch = non_tensor_batch
                self.check_consistency()

            def check_consistency(self):
                for value in self.non_tensor_batch.values():
                    self.assert_array(value)

            @staticmethod
            def assert_array(value):
                if not isinstance(value, np.ndarray) or value.ndim != 1:
                    raise AssertionError("DataProto non_tensor_batch columns must be one-dimensional ndarrays")

            def repeat(self, repeat_times):
                return type(self)(
                    {key: np.repeat(value, repeat_times, axis=0) for key, value in self.non_tensor_batch.items()}
                )

            def select_idxs(self, indices):
                return type(self)({key: value[indices] for key, value in self.non_tensor_batch.items()})

            def reorder(self, indices):
                self.non_tensor_batch = {key: value[indices] for key, value in self.non_tensor_batch.items()}
                self.check_consistency()

        original = MinimalDataProto(
            {
                "prompt_identity": np.array(["train:10", "train:11"], dtype=object),
                "uid": np.array(["uid-a", "uid-b"], dtype=object),
            }
        )
        repeated = original.repeat(2)
        attach_stable_ids_to_batch(repeated, global_step=7, observer=BufferedObserver())
        filtered = repeated.select_idxs(np.array([3, 0, 2]))
        filtered.reorder(np.array([2, 0, 1]))

        self.assertEqual(filtered.non_tensor_batch["trajectory_id"].tolist(), [0, 1, 0])
        self.assertEqual(
            filtered.non_tensor_batch["group_id"].tolist(),
            [repeated.non_tensor_batch["group_id"][2], repeated.non_tensor_batch["group_id"][3], repeated.non_tensor_batch["group_id"][0]],
        )

    def test_ocp_facts_emit_winner_identity_without_tensors(self) -> None:
        facts = OCPSelectionFacts(
            group_id="stable-group",
            winner_local_index=3,
            trajectory_id=1,
            mean_old_log_prob=-0.25,
        )

        self.assertEqual(
            ocp_selection_event(facts),
            {
                "event_type": "ocp_selection",
                "group_id": "stable-group",
                "winner_local_index": 3,
                "trajectory_id": 1,
                "mean_old_log_prob": -0.25,
            },
        )

    def test_launcher_exposes_outer_adapter_to_upstream_working_directory(self) -> None:
        from latent_grpo_runner.config import load_config
        from latent_grpo_runner.distributed import launch

        config = load_config(ROOT / "configs" / "3gpu-low.yaml", workspace_root=ROOT)
        seen: dict[str, object] = {}

        def fake_run(command: tuple[str, ...], **kwargs: object) -> int:
            seen.update(kwargs)
            return 0

        launch(
            config,
            run_command=fake_run,
            environment={"LATENT_GRPO_OBSERVER_ENABLED": "1", "PYTHONPATH": "/existing"},
        )

        python_paths = str(seen["env"]["PYTHONPATH"]).split(os.pathsep)
        self.assertEqual(python_paths[0], str(ROOT))
        self.assertEqual(seen["cwd"], ROOT / "Latent-GRPO" / "verl-0.4.x")

        seen.clear()
        config = config.with_runtime_overrides(metrics_enabled=False)
        launch(config, run_command=fake_run, environment={"PYTHONPATH": "/existing"})
        self.assertEqual(seen["env"]["PYTHONPATH"], "/existing")
        self.assertEqual(seen["env"]["LATENT_GRPO_OBSERVER_ENABLED"], "0")

    def test_component_stats_are_detached_scalar_sufficient_statistics(self) -> None:
        from latent_grpo_runner.upstream_adapter import build_component_sufficient_stats

        stats = build_component_sufficient_stats(
            margins=[-0.25, 0.0, 0.5, float("nan")],
            valid_mask=[True, True, False, True],
            flip_mask=[True, False, True, False],
            near_zero_threshold=0.01,
        )

        self.assertEqual(
            stats,
            {
                "sum": -0.25,
                "sum_sq": 0.0625,
                "count": 2,
                "nan_count": 1,
                "masked_count": 1,
                "min": -0.25,
                "negative_count": 1,
                "near_zero_count": 1,
                "flipgrad_trigger_count": 1,
            },
        )
        self.assertTrue(all(isinstance(value, (int, float)) for value in stats.values()))


if __name__ == "__main__":
    unittest.main()
