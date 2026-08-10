import ast
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "Latent_GRPO_Kaggle_2xT4_30_Metric_Runtime_Validation.ipynb"
EXPECTED_COMMIT = "1ab101c85ef1a75d1ed99011edbd0ca32ca68b87"

CORE_METRICS = (
    "train/policy_loss",
    "train/entropy",
    "train/kl",
    "train/clip_fraction",
    "train/importance_ratio_mean",
    "train/importance_ratio_std",
    "train/response_length",
    "train/latent_length",
    "train/generated_token_count",
    "train/step_time",
    "mixture/effective_k_noisy",
    "mixture/top1_weight_noisy",
    "mask/zero_advantage_rate",
    "signal/reward_mean",
    "signal/reward_std",
    "signal/advantage_std",
    "support/retention_rate",
    "support/top1_retention_rate",
    "onesided/delta_mean",
    "onesided/delta_std",
    "onesided/delta_p05",
    "onesided/delta_min",
    "onesided/delta_negative_rate",
    "onesided/delta_near_zero_rate",
    "onesided/flipgrad_rate",
    "credit/top1_share",
    "credit/effective_k",
    "credit/weight_credit_spearman",
    "credit/surrogate_alignment_rate",
)


class Kaggle30MetricNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.code = "\n".join(
            "".join(cell.get("source", []))
            if isinstance(cell.get("source", []), list)
            else cell.get("source", "")
            for cell in cls.notebook["cells"]
            if cell["cell_type"] == "code"
        )
        cls.all_source = "\n".join(
            "".join(cell.get("source", []))
            if isinstance(cell.get("source", []), list)
            else cell.get("source", "")
            for cell in cls.notebook["cells"]
        )

    def test_notebook_has_29_core_metrics_plus_raw_extension(self) -> None:
        self.assertEqual(len(CORE_METRICS), 29)
        for metric in CORE_METRICS:
            self.assertIn(metric, self.code)
        self.assertIn("train/raw_generated_token_count", self.code)
        self.assertIn("assert len(CAPTURE_METRICS) == 30", self.code)
        self.assertIn("CORE METRICS:", self.code)
        self.assertIn("KAGGLE_30_CAPTURE_GATE:", self.code)

    def test_notebook_is_pinned_and_uses_the_formal_dual_t4_runner(self) -> None:
        self.assertIn(EXPECTED_COMMIT, self.code)
        self.assertNotIn("__EXPECTED_COMMIT__", self.all_source)
        self.assertIn("tools/run_kaggle_t4_30_metric_validation.py", self.code)
        self.assertIn("configs/kaggle-t4-30-metric.yaml", self.code)
        for flag in ("--model-path", "--train-path", "--val-path", "--output-root"):
            self.assertIn(flag, self.code)

    def test_notebook_fails_closed_over_runtime_and_non_pollution_evidence(self) -> None:
        for gate in (
            "GIT_IDENTITY_GATE",
            "CURRENT_GIT_STAGE34_RUNTIME_ENTRYPOINT_GATE",
            "KAGGLE_DUAL_T4_HARDWARE_GATE",
            "REAL_STAGE123_RUNTIME_GATE",
            "STAGE123_PASSIVE_INSTRUMENTATION_GATE",
            "REAL_STAGE4_CHECKPOINT_PROBE_GATE",
        ):
            self.assertIn(gate, self.code)
        for evidence in (
            "stage123_non_pollution.json",
            "stage4_state_preservation.json",
            "parameters_changed_by_probe",
            "optimizer_state_changed",
            "training_grad_polluted",
            "cpu_rng_restored",
            "cuda_rng_restored",
            "module_mode_restored",
        ):
            self.assertIn(evidence, self.code)

    def test_notebook_is_noninteractive_and_does_not_patch_repo_source(self) -> None:
        for forbidden in (
            "input(",
            "getpass(",
            "CONFIG_PY.write_text",
            "PROFILE_YAML.write_text",
            "FSDP_SGLANG_PY.write_text",
            "_get_or_create_event_loop",
        ):
            self.assertNotIn(forbidden, self.code)

    def test_all_code_cells_are_valid_python(self) -> None:
        for index, cell in enumerate(self.notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = cell.get("source", "")
            if isinstance(source, list):
                source = "".join(source)
            try:
                ast.parse(source)
            except SyntaxError as error:
                self.fail(f"code cell {index} is not valid Python: {error}")


if __name__ == "__main__":
    unittest.main()
