#!/usr/bin/env python3
"""Build the pinned Kaggle dual-T4 29-core-plus-raw validation notebook."""

from __future__ import annotations

import os
from pathlib import Path
import re

import nbformat


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Latent_GRPO_Kaggle_2xT4_30_Metric_Runtime_Validation.ipynb"
EXPECTED_COMMIT = os.environ.get("LATENT_GRPO_EXPECTED_COMMIT", "").strip()
if re.fullmatch(r"[0-9a-f]{40}", EXPECTED_COMMIT) is None:
    raise SystemExit("set LATENT_GRPO_EXPECTED_COMMIT to the published 40-character Git SHA")


def markdown(source: str):
    return nbformat.v4.new_markdown_cell(source.replace("__EXPECTED_COMMIT__", EXPECTED_COMMIT).strip() + "\n")


def code(source: str):
    return nbformat.v4.new_code_cell(source.replace("__EXPECTED_COMMIT__", EXPECTED_COMMIT).strip() + "\n")


cells = [
    markdown(
        """
# Latent-GRPO Kaggle 2xT4 30-Metric Runtime Validation

Run **Run All** in a fresh Kaggle notebook with Internet enabled and exactly two NVIDIA T4 GPUs.

This notebook validates repository commit `__EXPECTED_COMMIT__`. The checked-out Git tree is the only source of truth. The notebook never patches source, reconstructs metrics, calls `loss.backward()`, or calls `optimizer.step()`.

The run validates 29 core metrics plus `train/raw_generated_token_count`, for 30 captured metrics. Missing formal schema, profile, runtime hook, metric rows, or state-preservation evidence produces `BLOCKED`/`NOT_REACHED`, never an inferred pass. Kaggle is an engineering validation platform; the final 3-GPU gate remains `TARGET_RUNTIME_REQUIRED`.
"""
    ),
    markdown("## 01. Constants"),
    code(
        """
from pathlib import Path

REPO_URL = "https://github.com/fengjixing-stupid/Latent-GRPO.git"
EXPECTED_COMMIT = "__EXPECTED_COMMIT__"

WORKING_ROOT = Path("/kaggle/working")
REPO_DIR = WORKING_ROOT / "Latent-GRPO"
VENV_DIR = WORKING_ROOT / "latent-t4-cu124"
VENV_PYTHON = VENV_DIR / "bin/python"
OUTPUT_ROOT = WORKING_ROOT / "latent-grpo-29-metric-validation"
RUNTIME_REPORT = WORKING_ROOT / "latent-grpo-29-metric-compatibility.json"
RUNTIME_IMPORT_REPORT = WORKING_ROOT / "latent-grpo-29-metric-runtime-imports.json"
STATE_PRESERVATION_REPORT = OUTPUT_ROOT / "stage4_state_preservation.json"
STAGE123_NON_POLLUTION_REPORT = OUTPUT_ROOT / "stage123_non_pollution.json"

MODEL_PATH = Path("/kaggle/input/models/fengjixing/llama3-2-1b-instruct-latent-sft-top10/pytorch/latent-sft/1/LLaMA3.2-1B-Instruct-Latent-SFT-Top10")
TRAIN_PATH = Path("/kaggle/input/datasets/fengjixing/latent-rl-data/data/GSM8k-Aug-oss-dup-all.parquet")
VAL_PATH = Path("/kaggle/input/datasets/fengjixing/latent-rl-data/data/GSM8k-Aug-test.parquet")
EXPECTED_TRAIN_SHA256 = "3766e3a83cd82ddd686392d8bc6ef6f262821490a09b694b2caed44f1a482501"
EXPECTED_VAL_SHA256 = "fd36cfb91155f2fd3b53ae3b0377543b9f92dfec20ada81835b4e39902689add"

FORMAL_INSTALLER = Path("tools/install_kaggle_t4_runtime.sh")
FORMAL_COMPATIBILITY_PROBE = Path("tools/probe_kaggle_p1_t4_compatibility.py")
FORMAL_RUNTIME_RUNNER = Path("tools/run_kaggle_t4_30_metric_validation.py")

METRIC_SPECS = [
    {'metric': 'train/policy_loss', 'stage': 'Stage 1', 'family': 'loss', 'source_table': 'train_step_metrics', 'invariant': 'finite'},
    {'metric': 'train/entropy', 'stage': 'Stage 1', 'family': 'entropy', 'source_table': 'train_step_metrics', 'invariant': 'finite_nonnegative'},
    {'metric': 'train/kl', 'stage': 'Stage 1', 'family': 'kl', 'source_table': 'train_step_metrics', 'invariant': 'finite'},
    {'metric': 'train/clip_fraction', 'stage': 'Stage 1', 'family': 'clip', 'source_table': 'train_step_metrics', 'invariant': 'rate'},
    {'metric': 'train/importance_ratio_mean', 'stage': 'Stage 1', 'family': 'importance_ratio', 'source_table': 'train_step_metrics', 'invariant': 'finite_positive'},
    {'metric': 'train/importance_ratio_std', 'stage': 'Stage 1', 'family': 'importance_ratio', 'source_table': 'train_step_metrics', 'invariant': 'finite_nonnegative'},
    {'metric': 'train/response_length', 'stage': 'Stage 1', 'family': 'length', 'source_table': 'train_step_metrics', 'invariant': 'finite_nonnegative'},
    {'metric': 'train/latent_length', 'stage': 'Stage 1', 'family': 'length', 'source_table': 'train_step_metrics', 'invariant': 'finite_nonnegative'},
    {'metric': 'train/generated_token_count', 'stage': 'Stage 1', 'family': 'tokens', 'source_table': 'train_step_metrics', 'invariant': 'finite_nonnegative'},
    {'metric': 'train/step_time', 'stage': 'Stage 1', 'family': 'time', 'source_table': 'train_step_metrics', 'invariant': 'finite_positive'},
    {'metric': 'mixture/effective_k_noisy', 'stage': 'Stage 2', 'family': 'mixture', 'source_table': 'train_step_metrics', 'invariant': 'effective_k'},
    {'metric': 'mixture/top1_weight_noisy', 'stage': 'Stage 2', 'family': 'mixture', 'source_table': 'train_step_metrics', 'invariant': 'rate'},
    {'metric': 'mask/zero_advantage_rate', 'stage': 'Stage 2', 'family': 'mask', 'source_table': 'train_step_metrics', 'invariant': 'rate'},
    {'metric': 'signal/reward_mean', 'stage': 'Stage 2', 'family': 'signal', 'source_table': 'train_step_metrics', 'invariant': 'finite'},
    {'metric': 'signal/reward_std', 'stage': 'Stage 2', 'family': 'signal', 'source_table': 'train_step_metrics', 'invariant': 'finite_nonnegative'},
    {'metric': 'signal/advantage_std', 'stage': 'Stage 2', 'family': 'signal', 'source_table': 'train_step_metrics', 'invariant': 'finite_nonnegative'},
    {'metric': 'support/retention_rate', 'stage': 'Stage 3', 'family': 'support', 'source_table': 'support_metrics', 'invariant': 'rate'},
    {'metric': 'support/top1_retention_rate', 'stage': 'Stage 3', 'family': 'support', 'source_table': 'support_metrics', 'invariant': 'rate'},
    {'metric': 'onesided/delta_mean', 'stage': 'Stage 4 One-sided', 'family': 'onesided', 'source_table': 'probe_metrics', 'invariant': 'finite'},
    {'metric': 'onesided/delta_std', 'stage': 'Stage 4 One-sided', 'family': 'onesided', 'source_table': 'probe_metrics', 'invariant': 'finite_nonnegative'},
    {'metric': 'onesided/delta_p05', 'stage': 'Stage 4 One-sided', 'family': 'onesided', 'source_table': 'probe_metrics', 'invariant': 'finite'},
    {'metric': 'onesided/delta_min', 'stage': 'Stage 4 One-sided', 'family': 'onesided', 'source_table': 'probe_metrics', 'invariant': 'finite'},
    {'metric': 'onesided/delta_negative_rate', 'stage': 'Stage 4 One-sided', 'family': 'onesided', 'source_table': 'probe_metrics', 'invariant': 'rate'},
    {'metric': 'onesided/delta_near_zero_rate', 'stage': 'Stage 4 One-sided', 'family': 'onesided', 'source_table': 'probe_metrics', 'invariant': 'rate'},
    {'metric': 'onesided/flipgrad_rate', 'stage': 'Stage 4 One-sided', 'family': 'onesided', 'source_table': 'probe_metrics', 'invariant': 'rate'},
    {'metric': 'credit/top1_share', 'stage': 'Stage 4 Credit', 'family': 'credit', 'source_table': 'probe_metrics', 'invariant': 'rate'},
    {'metric': 'credit/effective_k', 'stage': 'Stage 4 Credit', 'family': 'credit', 'source_table': 'probe_metrics', 'invariant': 'effective_k'},
    {'metric': 'credit/weight_credit_spearman', 'stage': 'Stage 4 Credit', 'family': 'credit', 'source_table': 'probe_metrics', 'invariant': 'correlation'},
    {'metric': 'credit/surrogate_alignment_rate', 'stage': 'Stage 4 Credit', 'family': 'credit', 'source_table': 'probe_metrics', 'invariant': 'rate'},
]

assert len(METRIC_SPECS) == 29
assert len({spec['metric'] for spec in METRIC_SPECS}) == 29
CAPTURE_METRICS = [spec['metric'] for spec in METRIC_SPECS] + ['train/raw_generated_token_count']
assert len(CAPTURE_METRICS) == 30
print("Expected commit:", EXPECTED_COMMIT)
print("Core metric count:", len(METRIC_SPECS))
print("Captured metric count:", len(CAPTURE_METRICS))
print("Output root:", OUTPUT_ROOT)
"""
    ),
    markdown("## 02. Fail-closed helpers"),
    code(
        """
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from collections import Counter

GATES = {}
BLOCKERS = []
RUNTIME_REACHED = {'Stage 1': False, 'Stage 2': False, 'Stage 3': False, 'Stage 4 One-sided': False, 'Stage 4 Credit': False}
CAPABILITY = {}
TABLE_ROWS = {}
ARTIFACT_READ_ERRORS = []

def mark_gate(name, passed, reason=None, *, blocking=True):
    status = "PASS" if passed else "BLOCKED"
    GATES[name] = {'status': status, 'reason': reason}
    print(f"{name}: {status}")
    if reason:
        print("  reason:", reason)
    if not passed and blocking:
        BLOCKERS.append(f"{name}:{reason or 'unspecified'}")
    return passed

def run_checked(command, *, cwd=None, env=None, capture=True):
    normalized = [str(item) for item in command]
    try:
        result = subprocess.run(
            normalized, cwd=None if cwd is None else str(cwd), env=env,
            text=True, capture_output=capture, check=False,
        )
    except OSError as error:
        message = f"command_start_error:{type(error).__name__}:{error}"
        print(message, file=sys.stderr)
        result = subprocess.CompletedProcess(normalized, 127, "", message)
    if capture and result.stdout:
        print(result.stdout[-8000:])
    if capture and result.stderr:
        print(result.stderr[-8000:], file=sys.stderr)
    return result

def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()

def safe_json(path):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        reason = f"json_read_error:{path}:{type(error).__name__}:{error}"
        ARTIFACT_READ_ERRORS.append(reason)
        print(reason, file=sys.stderr)
        return None

def invariant_ok(value, rule):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return False
    number = float(value)
    return {
        'finite': True,
        'finite_nonnegative': number >= 0,
        'finite_positive': number > 0,
        'rate': 0 <= number <= 1,
        'effective_k': number >= 1,
        'correlation': -1 <= number <= 1,
    }[rule]

def load_parquet_table(table_name):
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as error:
        reason = f"parquet_import_error:{type(error).__name__}:{error}"
        ARTIFACT_READ_ERRORS.append(reason)
        print(reason, file=sys.stderr)
        return []
    rows = []
    try:
        for part in sorted((OUTPUT_ROOT / table_name).glob("part-*.parquet")):
            rows.extend(pq.read_table(part).to_pylist())
    except Exception as error:
        reason = f"parquet_read_error:{table_name}:{type(error).__name__}:{error}"
        ARTIFACT_READ_ERRORS.append(reason)
        print(reason, file=sys.stderr)
        return []
    return rows

def display_rows(rows, columns):
    try:
        import pandas as pd
        from IPython.display import display
        display(pd.DataFrame(rows, columns=columns))
    except ModuleNotFoundError:
        print(json.dumps(rows, indent=2, default=str))
"""
    ),
    markdown("## 03. Clone and strict Git identity gate"),
    code(
        """
if REPO_DIR.exists():
    assert REPO_DIR.parent == WORKING_ROOT
    shutil.rmtree(REPO_DIR)

clone_result = run_checked(["git", "clone", REPO_URL, REPO_DIR])
fetch_result = run_checked(["git", "-C", REPO_DIR, "fetch", "--all"]) if clone_result.returncode == 0 else clone_result
checkout_result = run_checked(["git", "-C", REPO_DIR, "checkout", EXPECTED_COMMIT]) if fetch_result.returncode == 0 else fetch_result

head = ""
working_tree = "clone_failed"
if checkout_result.returncode == 0:
    head_result = run_checked(["git", "-C", REPO_DIR, "rev-parse", "HEAD"])
    status_result = run_checked(["git", "-C", REPO_DIR, "status", "--short"])
    head = head_result.stdout.strip()
    working_tree = status_result.stdout.strip()

git_ok = clone_result.returncode == fetch_result.returncode == checkout_result.returncode == 0
git_ok = git_ok and head == EXPECTED_COMMIT and working_tree == ""
mark_gate("GIT_IDENTITY_GATE", git_ok, None if git_ok else f"head={head!r}; status={working_tree!r}")
if git_ok:
    print("GIT_IDENTITY_GATE: PASS")
    print(head)
"""
    ),
    markdown("## 04. Current-Git schema and feature-definition gate"),
    code(
        """
REQUIRED_TABLES = {'support_metrics', 'support_benchmark_metrics', 'probe_metrics', 'probe_benchmark_metrics'}
REQUIRED_STAGE4_FIELDS = {
    'onesided/delta_mean', 'onesided/delta_std', 'onesided/delta_p05', 'onesided/delta_min',
    'onesided/delta_negative_rate', 'onesided/delta_near_zero_rate', 'onesided/flipgrad_rate',
    'credit/top1_share', 'credit/effective_k', 'credit/weight_credit_spearman',
    'credit/surrogate_alignment_rate',
}
REQUIRED_FEATURES = {'support_enabled', 'checkpoint_probe_enabled', 'credit_probe_enabled'}

schema_ok = GATES.get("GIT_IDENTITY_GATE", {}).get("status") == "PASS"
schema_reasons = []
schema_manifest = None
if schema_ok:
    sys.path.insert(0, str(REPO_DIR))
    try:
        from latent_grpo_runner.metrics.schemas import schema_manifest as current_schema_manifest
        schema_manifest = current_schema_manifest()
        stages = schema_manifest.get('stages', {})
        expected_status = {'stage1': 'enabled', 'stage2': 'enabled', 'stage3': 'enabled', 'stage4': 'checkpoint_probe_enabled'}
        for stage, status in expected_status.items():
            if stages.get(stage, {}).get('status') != status:
                schema_reasons.append(f"stage_status:{stage}={stages.get(stage)}")
        tables = schema_manifest.get('tables', {})
        missing_tables = sorted(REQUIRED_TABLES - set(tables))
        if missing_tables:
            schema_reasons.append(f"missing_tables:{missing_tables}")
        probe_names = {field.get('name') for field in tables.get('probe_metrics', {}).get('fields', [])}
        missing_stage4 = sorted(REQUIRED_STAGE4_FIELDS - probe_names)
        if missing_stage4:
            schema_reasons.append(f"missing_stage4_fields:{missing_stage4}")
        required_by_table = {}
        for metric_spec in METRIC_SPECS:
            required_by_table.setdefault(metric_spec['source_table'], set()).add(metric_spec['metric'])
        missing_core_metric_fields = {}
        for table_name, required_names in required_by_table.items():
            actual_names = {field.get('name') for field in tables.get(table_name, {}).get('fields', [])}
            missing_names = sorted(required_names - actual_names)
            if missing_names:
                missing_core_metric_fields[table_name] = missing_names
        if missing_core_metric_fields:
            schema_reasons.append(f"missing_core_metric_fields:{missing_core_metric_fields}")
        for table_name, table in tables.items():
            names = [field.get('name') for field in table.get('fields', [])]
            duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
            if duplicates:
                schema_reasons.append(f"duplicate_schema_fields:{table_name}:{duplicates}")
        config_source = (REPO_DIR / "latent_grpo_runner/config.py").read_text(encoding="utf-8")
        missing_features = sorted(name for name in REQUIRED_FEATURES if name not in config_source)
        if missing_features:
            schema_reasons.append(f"missing_feature_definitions:{missing_features}")
    except Exception as error:
        schema_reasons.append(f"schema_import_error:{type(error).__name__}:{error}")

schema_ok = schema_ok and not schema_reasons
mark_gate("CURRENT_GIT_29_METRIC_SCHEMA_GATE", schema_ok, None if schema_ok else "; ".join(schema_reasons))
if schema_ok:
    print("CURRENT_GIT_29_METRIC_SCHEMA_GATE: PASS")
"""
    ),
    markdown(
        """
## 05. Formal Stage 3/4 runtime-entrypoint gate

Schema helpers alone are not a GPU runtime path. This gate requires a two-T4 profile with all three feature gates enabled, a Stage 3 trainer hook, and a formal Stage 4 runner that executes the preserving checkpoint probe and emits `checkpoint_probe` rows. Missing capability is reported as `BLOCKED`; the notebook does not create or patch it.
"""
    ),
    code(
        """
capability_reasons = []
if GATES.get("GIT_IDENTITY_GATE", {}).get("status") == "PASS":
    trainer_path = REPO_DIR / "Latent-GRPO/verl-0.4.x/verl/trainer/ppo/ray_trainer.py"
    trainer_source = trainer_path.read_text(encoding="utf-8") if trainer_path.is_file() else ""
    CAPABILITY['stage3_trainer_hook'] = (
        'LATENT_GRPO_SUPPORT_ENABLED' in trainer_source
        and 'collect_support_metrics(' in trainer_source
        and trainer_source.count('compute_log_prob(batch)') == 1
    )

    two_t4_feature_profile = None
    try:
        import yaml
        for profile_path in sorted((REPO_DIR / "configs").glob("*.yaml")):
            profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
            features = profile.get('features', {})
            hardware = profile.get('target_hardware', {})
            if hardware.get('required_gpus') == 2 and all(features.get(name) is True for name in REQUIRED_FEATURES):
                two_t4_feature_profile = profile_path
                break
    except Exception as error:
        capability_reasons.append(f"profile_parse_error:{type(error).__name__}:{error}")
    CAPABILITY['two_t4_feature_profile'] = None if two_t4_feature_profile is None else str(two_t4_feature_profile)

    formal_stage4_sources = []
    formal_runner = REPO_DIR / FORMAL_RUNTIME_RUNNER
    runtime_sources = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in (
            formal_runner,
            REPO_DIR / "latent_grpo_runner/metrics/probe.py",
            REPO_DIR / "Latent-GRPO/verl-0.4.x/verl/workers/actor/dp_actor.py",
            trainer_path,
        )
        if path.is_file()
    )
    if (
        formal_runner.is_file()
        and all(flag in formal_runner.read_text(encoding="utf-8") for flag in ('--model-path', '--train-path', '--val-path', '--output-root'))
        and 'collect_checkpoint_probe_packet' in runtime_sources
        and 'build_checkpoint_probe_event' in runtime_sources
        and 'credit_autograd_executed' in runtime_sources
        and 'emit("checkpoint_probe"' in runtime_sources
    ):
        formal_stage4_sources.append(formal_runner)
    CAPABILITY['formal_stage4_entrypoints'] = [str(path) for path in formal_stage4_sources]

    if not CAPABILITY['stage3_trainer_hook']:
        capability_reasons.append('missing_stage3_passive_trainer_hook')
    if two_t4_feature_profile is None:
        capability_reasons.append('missing_two_t4_profile_with_support_checkpoint_credit_enabled')
    if not formal_stage4_sources:
        capability_reasons.append('missing_formal_stage4_checkpoint_runner')
else:
    capability_reasons.append('git_identity_not_available')

CAPABILITY['ready'] = not capability_reasons
mark_gate(
    "CURRENT_GIT_STAGE34_RUNTIME_ENTRYPOINT_GATE",
    CAPABILITY['ready'],
    None if CAPABILITY['ready'] else "; ".join(capability_reasons),
)
print(json.dumps(CAPABILITY, indent=2))
"""
    ),
    markdown("## 06. Dual-T4 hardware identity gate"),
    code(
        """
hardware_result = run_checked([
    "nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader,nounits"
])
gpu_rows = [line.strip() for line in hardware_result.stdout.splitlines() if line.strip()]
hardware_identity_ok = hardware_result.returncode == 0 and len(gpu_rows) == 2
hardware_identity_ok = hardware_identity_ok and all('T4' in row and '7.5' in row for row in gpu_rows)
mark_gate(
    "KAGGLE_DUAL_T4_HARDWARE_IDENTITY_GATE",
    hardware_identity_ok,
    None if hardware_identity_ok else f"expected exactly 2x NVIDIA T4 cc7.5; observed={gpu_rows}",
)
print("GPU rows:", gpu_rows)
"""
    ),
    markdown("## 07. Install the current-Git isolated runtime"),
    code(
        """
setup_prerequisite_names = (
    'GIT_IDENTITY_GATE',
    'CURRENT_GIT_29_METRIC_SCHEMA_GATE',
    'CURRENT_GIT_STAGE34_RUNTIME_ENTRYPOINT_GATE',
    'KAGGLE_DUAL_T4_HARDWARE_IDENTITY_GATE',
)
setup_prerequisites = all(
    GATES.get(name, {}).get('status') == 'PASS'
    for name in setup_prerequisite_names
)
if setup_prerequisites:
    installer = REPO_DIR / FORMAL_INSTALLER
    install_result = run_checked(["bash", installer], cwd=REPO_DIR, capture=False)
    install_ok = install_result.returncode == 0 and VENV_PYTHON.is_file()
    mark_gate("CURRENT_GIT_RUNTIME_INSTALLER_GATE", install_ok, None if install_ok else f"exit={install_result.returncode}")
else:
    mark_gate("CURRENT_GIT_RUNTIME_INSTALLER_GATE", False, "prerequisite_gate_blocked")
"""
    ),
    markdown("## 08. Runtime import, CUDA, NCCL, FP16, and formal compatibility gates"),
    code(
        """
runtime_gate_code = r'''import importlib
import json
import platform
import torch

modules = ['triton', 'sglang', 'flashinfer', 'sgl_kernel', 'torch_memory_saver']
versions = {}
for name in modules:
    module = importlib.import_module(name)
    versions[name] = getattr(module, '__version__', 'unknown')

report = {
    'python': platform.python_version(),
    'torch': torch.__version__,
    'torch_cuda_build': torch.version.cuda,
    'cuda_available': torch.cuda.is_available(),
    'cuda_device_count': torch.cuda.device_count(),
    'gpu_names': [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
    'compute_capabilities': [list(torch.cuda.get_device_capability(index)) for index in range(torch.cuda.device_count())],
    'nccl_available': torch.distributed.is_nccl_available(),
    'fp16_supported': all(torch.cuda.get_device_capability(index) >= (5, 3) for index in range(torch.cuda.device_count())),
    'module_versions': versions,
}
print(json.dumps(report, sort_keys=True))'''

runtime_ready = GATES.get('CURRENT_GIT_RUNTIME_INSTALLER_GATE', {}).get('status') == 'PASS'
runtime_import_report = None
if runtime_ready:
    import_result = run_checked([VENV_PYTHON, "-c", runtime_gate_code], cwd=REPO_DIR)
    try:
        runtime_import_report = json.loads(import_result.stdout.strip().splitlines()[-1])
    except Exception:
        runtime_import_report = None
    runtime_ok = import_result.returncode == 0 and runtime_import_report is not None
    runtime_ok = runtime_ok and runtime_import_report['python'].startswith('3.10.')
    runtime_ok = runtime_ok and runtime_import_report['torch_cuda_build'] == '12.4'
    runtime_ok = runtime_ok and runtime_import_report['cuda_available'] is True
    runtime_ok = runtime_ok and runtime_import_report['cuda_device_count'] == 2
    runtime_ok = runtime_ok and all('T4' in name for name in runtime_import_report['gpu_names'])
    runtime_ok = runtime_ok and runtime_import_report['compute_capabilities'] == [[7, 5], [7, 5]]
    runtime_ok = runtime_ok and runtime_import_report['nccl_available'] is True
    runtime_ok = runtime_ok and runtime_import_report['fp16_supported'] is True
    if runtime_import_report is not None:
        RUNTIME_IMPORT_REPORT.parent.mkdir(parents=True, exist_ok=True)
        RUNTIME_IMPORT_REPORT.write_text(json.dumps(runtime_import_report, indent=2) + "\\n", encoding="utf-8")
    mark_gate("KAGGLE_DUAL_T4_HARDWARE_GATE", runtime_ok, None if runtime_ok else "runtime_import_or_cuda_contract_failed")
    if runtime_ok:
        print("KAGGLE_DUAL_T4_HARDWARE_GATE: PASS")

    compatibility_result = run_checked([
        VENV_PYTHON, REPO_DIR / FORMAL_COMPATIBILITY_PROBE, "--report", RUNTIME_REPORT
    ], cwd=REPO_DIR)
    compatibility_report = safe_json(RUNTIME_REPORT)
    formal_ok = compatibility_result.returncode == 0 and compatibility_report is not None
    formal_ok = formal_ok and compatibility_report.get('commit') == EXPECTED_COMMIT
    formal_ok = formal_ok and compatibility_report.get('status') == 'READY_FOR_DATA'
    formal_ok = formal_ok and compatibility_report.get('blockers') == []
    formal_ok = formal_ok and compatibility_report.get('training_started') is False
    mark_gate("KAGGLE_DUAL_T4_FORMAL_GATE", formal_ok, None if formal_ok else "formal_compatibility_probe_failed")
    if formal_ok:
        print("KAGGLE_DUAL_T4_FORMAL_GATE: PASS")
else:
    mark_gate("KAGGLE_DUAL_T4_HARDWARE_GATE", False, "runtime_installer_not_reached")
    mark_gate("KAGGLE_DUAL_T4_FORMAL_GATE", False, "runtime_installer_not_reached")
"""
    ),
    markdown("## 09. Author model and real GSM8K asset gates"),
    code(
        """
model_config_assets = [MODEL_PATH / 'config.json']
tokenizer_assets = list(MODEL_PATH.glob('tokenizer*')) + list(MODEL_PATH.glob('special_tokens_map.json'))
weight_files = sorted(list(MODEL_PATH.glob('*.safetensors')) + list(MODEL_PATH.glob('pytorch_model*.bin')))
model_ok = MODEL_PATH.is_dir()
model_ok = model_ok and all(path.is_file() and path.stat().st_size > 0 for path in model_config_assets)
model_ok = model_ok and bool(tokenizer_assets) and all(path.stat().st_size > 0 for path in tokenizer_assets)
model_ok = model_ok and bool(weight_files) and all(path.stat().st_size > 0 for path in weight_files)
mark_gate("AUTHOR_SFT_MODEL_ASSET", model_ok, None if model_ok else "missing_config_tokenizer_or_nonempty_weights")

data_ok = TRAIN_PATH.is_file() and VAL_PATH.is_file()
data_reason = None
if data_ok:
    train_sha = sha256_file(TRAIN_PATH)
    val_sha = sha256_file(VAL_PATH)
    data_ok = TRAIN_PATH.stat().st_size > 0 and VAL_PATH.stat().st_size > 0
    data_ok = data_ok and train_sha == EXPECTED_TRAIN_SHA256 and val_sha == EXPECTED_VAL_SHA256
    try:
        import pyarrow.parquet as pq
        required_columns = {'prompt', 'data_source', 'reward_model'}
        for path in (TRAIN_PATH, VAL_PATH):
            parquet = pq.ParquetFile(path)
            data_ok = data_ok and parquet.metadata.num_rows > 0
            data_ok = data_ok and required_columns.issubset(set(parquet.schema_arrow.names))
    except Exception as error:
        data_ok = False
        data_reason = f"parquet_validation_error:{type(error).__name__}:{error}"
else:
    data_reason = "missing_train_or_validation_parquet"
mark_gate("GSM8K_DATA_IDENTITY_GATE", data_ok, None if data_ok else data_reason or "size_sha256_or_schema_mismatch")
"""
    ),
    markdown("## 10. Latent-end tokenizer gate"),
    code(
        """
tokenizer_gate_code = r'''import json
import sys
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(sys.argv[1], trust_remote_code=False)
marker_ids = tokenizer.encode('</think>', add_special_tokens=False)
print(json.dumps({'marker': '</think>', 'marker_ids': marker_ids, 'first_id': marker_ids[0] if marker_ids else None}))
raise SystemExit(0 if marker_ids and marker_ids[0] == 524 else 2)'''

if VENV_PYTHON.is_file() and GATES.get('AUTHOR_SFT_MODEL_ASSET', {}).get('status') == 'PASS':
    tokenizer_result = run_checked([VENV_PYTHON, "-c", tokenizer_gate_code, MODEL_PATH], cwd=REPO_DIR)
    tokenizer_ok = tokenizer_result.returncode == 0
    mark_gate("LATENT_END_MARKER_GATE", tokenizer_ok, None if tokenizer_ok else "expected first marker token id 524")
else:
    mark_gate("LATENT_END_MARKER_GATE", False, "runtime_or_model_not_available")
"""
    ),
    markdown("## 11. No-training config and feature dry-run gate"),
    code(
        """
base_dry_run = [
    VENV_PYTHON, REPO_DIR / "train_latent_grpo.py",
    "--config", REPO_DIR / "configs/kaggle-t4-30-metric.yaml",
    "--profile-name", "kaggle-t4-30-metric",
    "--model-path", MODEL_PATH,
    "--train-files", TRAIN_PATH,
    "--val-files", VAL_PATH,
    "--output-root", OUTPUT_ROOT,
    "--dry-run",
]
if VENV_PYTHON.is_file():
    base_result = run_checked(base_dry_run, cwd=REPO_DIR)
    config_ok = base_result.returncode == 0 and CAPABILITY.get('ready') is True
    config_reason = None if config_ok else (
        f"base_exit={base_result.returncode}; formal_stage34_ready={CAPABILITY.get('ready')}"
    )
    mark_gate("CONFIG_DRY_RUN_GATE", config_ok, config_reason)
    if config_ok:
        print("CONFIG_DRY_RUN_GATE: PASS")
else:
    mark_gate("CONFIG_DRY_RUN_GATE", False, "runtime_python_not_available")
"""
    ),
    markdown("## 12. Clean notebook-owned output root"),
    code(
        """
assert OUTPUT_ROOT.parent == WORKING_ROOT
assert OUTPUT_ROOT not in {REPO_DIR, VENV_DIR, MODEL_PATH, TRAIN_PATH, VAL_PATH}
clean_ok = True
clean_reason = None
try:
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
except OSError as error:
    clean_ok = False
    clean_reason = f"clean_output_error:{type(error).__name__}:{error}"
mark_gate("CLEAN_VALIDATION_OUTPUT_GATE", clean_ok, clean_reason)
"""
    ),
    markdown(
        """
## 13. Real Stage 1/2/3/4 execution

One formal runner performs the real Ray/FSDP/NCCL/SGLang step, passive Stage 1-3 collection, the checkpoint-only Stage 4 probe, and repository validation. The notebook does not run a second training path or recompute metric values.
"""
    ),
    code(
        """
execution_gate_names = (
    'GIT_IDENTITY_GATE', 'CURRENT_GIT_29_METRIC_SCHEMA_GATE',
    'CURRENT_GIT_STAGE34_RUNTIME_ENTRYPOINT_GATE', 'KAGGLE_DUAL_T4_HARDWARE_GATE',
    'KAGGLE_DUAL_T4_FORMAL_GATE', 'AUTHOR_SFT_MODEL_ASSET', 'GSM8K_DATA_IDENTITY_GATE',
    'LATENT_END_MARKER_GATE', 'CONFIG_DRY_RUN_GATE', 'CLEAN_VALIDATION_OUTPUT_GATE',
)
execution_ready = all(GATES.get(name, {}).get('status') == 'PASS' for name in execution_gate_names)

if execution_ready:
    formal_runtime_result = run_checked([
        VENV_PYTHON, REPO_DIR / FORMAL_RUNTIME_RUNNER,
        "--model-path", MODEL_PATH, "--train-path", TRAIN_PATH,
        "--val-path", VAL_PATH, "--output-root", OUTPUT_ROOT,
    ], cwd=REPO_DIR, capture=False)
    formal_runtime_ok = formal_runtime_result.returncode == 0
    mark_gate("REAL_STAGE123_RUNTIME_GATE", formal_runtime_ok, None if formal_runtime_ok else f"formal_runner_exit={formal_runtime_result.returncode}")
else:
    formal_runtime_ok = False
    mark_gate("REAL_STAGE123_RUNTIME_GATE", False, "prerequisite_or_formal_entrypoint_blocked", blocking=False)
"""
    ),
    markdown("## 14. Real Stage 4 checkpoint probe"),
    code(
        """
if execution_ready:
    stage4_ok = formal_runtime_ok and STATE_PRESERVATION_REPORT.is_file()
    mark_gate("REAL_STAGE4_CHECKPOINT_PROBE_GATE", stage4_ok, None if stage4_ok else "formal_runtime_or_state_report_failed")
else:
    mark_gate("REAL_STAGE4_CHECKPOINT_PROBE_GATE", False, "prerequisite_or_formal_entrypoint_blocked", blocking=False)
"""
    ),
    markdown("## 15. Load formal observer/sink/probe tables"),
    code(
        """
for table_name in ('train_step_metrics', 'support_metrics', 'support_benchmark_metrics', 'probe_metrics', 'probe_benchmark_metrics'):
    TABLE_ROWS[table_name] = load_parquet_table(table_name)
    print(table_name, "rows=", len(TABLE_ROWS[table_name]))

RUNTIME_REACHED['Stage 1'] = bool(TABLE_ROWS['train_step_metrics'])
RUNTIME_REACHED['Stage 2'] = bool(TABLE_ROWS['train_step_metrics'])
RUNTIME_REACHED['Stage 3'] = bool(TABLE_ROWS['support_metrics'])
RUNTIME_REACHED['Stage 4 One-sided'] = bool(TABLE_ROWS['probe_metrics'])
RUNTIME_REACHED['Stage 4 Credit'] = bool(TABLE_ROWS['probe_metrics'])
"""
    ),
    markdown("## 16. Validate each of the 29 formal metrics"),
    code(
        """
def availability_fields(metric):
    if metric.startswith('support/'):
        return 'support_available', 'support_unavailable_reason'
    if metric.startswith('onesided/'):
        return 'onesided_available', 'onesided_unavailable_reason'
    if metric in {'credit/top1_share', 'credit/effective_k'}:
        return 'credit_concentration_available', 'credit_concentration_unavailable_reason'
    if metric == 'credit/weight_credit_spearman':
        return 'credit/weight_credit_spearman__available', 'credit/weight_credit_spearman__unavailable_reason'
    if metric == 'credit/surrogate_alignment_rate':
        return 'credit/surrogate_alignment_rate__available', 'credit/surrogate_alignment_rate__unavailable_reason'
    return f'{metric}__available', f'{metric}__unavailable_reason'

def stage_specific_contract(spec, rows):
    stage = spec['stage']
    if stage == 'Stage 3':
        benchmark = TABLE_ROWS['support_benchmark_metrics']
        return (
            bool(benchmark)
            and all(row.get('support/effective_position_count', 0) > 0 for row in rows)
            and all(row.get('support_available') is True for row in rows)
            and any(row.get('support_benchmark/total_effective_position_count', 0) > 0 for row in benchmark)
        )
    if stage == 'Stage 4 One-sided':
        return all(
            row.get('onesided/delta_count', 0) > 0
            and row.get('valid_flipgrad_denominator', 0) > 0
            and isinstance(row.get('onesided_near_zero_threshold'), (int, float))
            and bool(row.get('onesided_definition_version'))
            for row in rows
        )
    if stage == 'Stage 4 Credit':
        benchmarks = TABLE_ROWS['probe_benchmark_metrics']
        return bool(benchmarks) and any(row.get('credit_autograd_executed') is True for row in benchmarks)
    return True

canonical_rows = []
for spec in METRIC_SPECS:
    metric = spec['metric']
    source_rows = TABLE_ROWS.get(spec['source_table'], [])
    available_key, reason_key = availability_fields(metric)
    available_rows = [row for row in source_rows if row.get(available_key) is True and row.get(metric) is not None]
    values = [row.get(metric) for row in available_rows]
    runtime_reached = RUNTIME_REACHED[spec['stage']]
    invariant_passed = bool(values) and all(invariant_ok(value, spec['invariant']) for value in values)
    contract_passed = bool(available_rows) and stage_specific_contract(spec, available_rows)

    if available_rows and invariant_passed and contract_passed:
        status = 'PASS'
        unavailable_reason = None
    elif runtime_reached:
        status = 'BLOCKED'
        reasons = [row.get(reason_key) for row in source_rows if row.get(reason_key)]
        unavailable_reason = '; '.join(sorted(set(reasons))) or 'availability_or_invariant_contract_failed'
    else:
        status = 'NOT_REACHED'
        unavailable_reason = 'formal_runtime_not_reached:' + '; '.join(BLOCKERS[:4])

    displayed_value = None
    if len(values) == 1:
        displayed_value = values[0]
    elif values:
        displayed_value = values[:8]
    canonical_rows.append({
        'metric': metric,
        'stage': spec['stage'],
        'family': spec['family'],
        'value': displayed_value,
        'available': bool(available_rows),
        'source_table': spec['source_table'],
        'runtime_reached': runtime_reached,
        'invariant_check': f"{spec['invariant']}:{'PASS' if invariant_passed and contract_passed else 'BLOCKED'}",
        'status': status,
        'unavailable_reason': unavailable_reason,
    })

canonical_columns = ['metric', 'stage', 'family', 'value', 'available', 'source_table', 'runtime_reached', 'invariant_check', 'status', 'unavailable_reason']
assert len(canonical_rows) == 29
assert {row['status'] for row in canonical_rows}.issubset({'PASS', 'BLOCKED', 'NOT_REACHED'})
display_rows(canonical_rows, canonical_columns)
"""
    ),
    markdown("## 17. Stage 1/2/3 passive instrumentation safety gate"),
    code(
        """
trainer_source = (REPO_DIR / "Latent-GRPO/verl-0.4.x/verl/trainer/ppo/ray_trainer.py").read_text(encoding="utf-8") if REPO_DIR.is_dir() else ""
support_source = (REPO_DIR / "latent_grpo_runner/metrics/support.py").read_text(encoding="utf-8") if REPO_DIR.is_dir() else ""
observer_sources = "".join(
    path.read_text(encoding="utf-8", errors="replace")
    for path in (REPO_DIR / "latent_grpo_runner/metrics").glob("*.py")
) if REPO_DIR.is_dir() else ""

stage13_static_contract = (
    CAPABILITY.get('stage3_trainer_hook') is True
    and 'torch.autograd.grad' not in support_source
    and 'loss.backward(' not in support_source
    and 'optimizer.step(' not in support_source
    and 'loss.backward(' not in observer_sources
    and 'optimizer.step(' not in observer_sources
)
stage13_runtime_reached = RUNTIME_REACHED['Stage 1'] and RUNTIME_REACHED['Stage 3']
stage123_report = safe_json(STAGE123_NON_POLLUTION_REPORT) or {}
stage123_required_evidence = {
    'extra_model_forward': False,
    'extra_loss_backward': False,
    'extra_optimizer_step': False,
    'parameters_changed_by_observer': False,
    'grads_changed_by_observer': False,
    'rollout_changed_by_observer': False,
    'filtered_batch_changed_by_observer': False,
    'rng_consumed_by_observer': False,
}
stage123_report_passed = bool(stage123_report) and all(
    stage123_report.get(key) is expected for key, expected in stage123_required_evidence.items()
)
stage13_safety_evidence = stage13_static_contract and stage13_runtime_reached and stage123_report_passed
mark_gate(
    "STAGE123_PASSIVE_INSTRUMENTATION_GATE",
    stage13_safety_evidence,
    None if stage13_safety_evidence else "missing_or_failed_formal_stage123_non_pollution_evidence",
)
"""
    ),
    markdown("## 18. Stage 4 state-preservation and non-pollution gate"),
    code(
        """
state_report = safe_json(STATE_PRESERVATION_REPORT) or {}
probe_benchmarks = TABLE_ROWS.get('probe_benchmark_metrics', [])
credit_autograd_executed = any(row.get('credit_autograd_executed') is True for row in probe_benchmarks)
probe_rng_restore_succeeded = bool(probe_benchmarks) and all(row.get('probe_rng_restore_succeeded') is True for row in probe_benchmarks)
no_extra_backward = stage123_report.get('extra_loss_backward') is False and state_report.get('extra_loss_backward') is False
no_extra_optimizer_step = stage123_report.get('extra_optimizer_step') is False and state_report.get('extra_optimizer_step') is False

safety_specs = [
    ('extra Stage1-3 model forward', 'NO', stage13_safety_evidence, 'NO' if stage13_safety_evidence else 'UNPROVEN'),
    ('extra loss.backward', 'NO', no_extra_backward, 'NO' if no_extra_backward else 'UNPROVEN'),
    ('extra optimizer.step', 'NO', no_extra_optimizer_step, 'NO' if no_extra_optimizer_step else 'UNPROVEN'),
    ('parameters changed by probe', 'NO', state_report.get('parameters_changed_by_probe') is False, state_report.get('parameters_changed_by_probe', 'UNPROVEN')),
    ('optimizer state changed', 'NO', state_report.get('optimizer_state_changed') is False, state_report.get('optimizer_state_changed', 'UNPROVEN')),
    ('training grad polluted', 'NO', state_report.get('training_grad_polluted') is False, state_report.get('training_grad_polluted', 'UNPROVEN')),
    ('CPU RNG restored', 'YES', state_report.get('cpu_rng_restored') is True, state_report.get('cpu_rng_restored', 'UNPROVEN')),
    ('CUDA RNG restored', 'YES', state_report.get('cuda_rng_restored') is True, state_report.get('cuda_rng_restored', 'UNPROVEN')),
    ('Python RNG restored', 'YES', state_report.get('python_rng_restored') is True, state_report.get('python_rng_restored', 'UNPROVEN')),
    ('NumPy RNG restored', 'YES/N/A', state_report.get('numpy_rng_restored') in {True, 'NOT_APPLICABLE'}, state_report.get('numpy_rng_restored', 'UNPROVEN')),
    ('module mode restored', 'YES', state_report.get('module_mode_restored') is True, state_report.get('module_mode_restored', 'UNPROVEN')),
]
safety_rows = [
    {'check': check, 'expected': expected, 'observed': observed, 'status': 'PASS' if passed else 'BLOCKED'}
    for check, expected, passed, observed in safety_specs
]

print("CREDIT_AUTOGRAD_EXECUTED:", "YES" if credit_autograd_executed else "NO/NOT_REACHED")
print("EXTRA_MODEL_FORWARD_FOR_STAGE1_3:", "NO" if stage13_safety_evidence else "UNPROVEN")
print("EXTRA_LOSS_BACKWARD:", "NO" if no_extra_backward else "UNPROVEN")
print("EXTRA_OPTIMIZER_STEP:", "NO" if no_extra_optimizer_step else "UNPROVEN")
print("STAGE123_PARAMETERS_CHANGED_BY_OBSERVER:", "NO" if stage123_report.get('parameters_changed_by_observer') is False else "UNPROVEN")
print("STAGE123_GRADS_CHANGED_BY_OBSERVER:", "NO" if stage123_report.get('grads_changed_by_observer') is False else "UNPROVEN")
print("STAGE123_ROLLOUT_CHANGED_BY_OBSERVER:", "NO" if stage123_report.get('rollout_changed_by_observer') is False else "UNPROVEN")
print("STAGE123_FILTERED_BATCH_CHANGED_BY_OBSERVER:", "NO" if stage123_report.get('filtered_batch_changed_by_observer') is False else "UNPROVEN")
print("STAGE123_RNG_CONSUMED_BY_OBSERVER:", "NO" if stage123_report.get('rng_consumed_by_observer') is False else "UNPROVEN")
print("PARAMETERS_CHANGED_BY_PROBE:", "NO" if state_report.get('parameters_changed_by_probe') is False else "UNPROVEN")
print("OPTIMIZER_STATE_CHANGED_BY_PROBE:", "NO" if state_report.get('optimizer_state_changed') is False else "UNPROVEN")
print("TRAINING_GRAD_POLLUTION:", "NO" if state_report.get('training_grad_polluted') is False else "UNPROVEN")
print("CPU_RNG_RESTORED:", "PASS" if state_report.get('cpu_rng_restored') is True else "BLOCKED")
print("CUDA_RNG_RESTORED:", "PASS" if state_report.get('cuda_rng_restored') is True else "BLOCKED")
print("PYTHON_RNG_RESTORED:", "PASS" if state_report.get('python_rng_restored') is True else "BLOCKED")
print(
    "NUMPY_RNG_RESTORED:",
    "NOT_APPLICABLE" if state_report.get('numpy_rng_restored') == 'NOT_APPLICABLE'
    else ("PASS" if state_report.get('numpy_rng_restored') is True else "BLOCKED"),
)
print("MODULE_MODE_RESTORED:", "PASS" if state_report.get('module_mode_restored') is True else "BLOCKED")
print("PROBE_RNG_RESTORE_SUCCEEDED:", "PASS" if probe_rng_restore_succeeded else "BLOCKED")
display_rows(safety_rows, ['check', 'expected', 'observed', 'status'])
"""
    ),
    markdown("## 19. raw_generated_token_count extension validation"),
    code(
        """
raw_rows = [
    row for row in TABLE_ROWS.get('train_step_metrics', [])
    if row.get('train/raw_generated_token_count__available') is True
    and row.get('train/generated_token_count__available') is True
]
raw_runtime_reached = bool(TABLE_ROWS.get('train_step_metrics'))
raw_ok = bool(raw_rows) and all(
    row['train/raw_generated_token_count'] >= row['train/generated_token_count']
    for row in raw_rows
)
raw_status = 'PASS' if raw_ok else ('BLOCKED' if raw_runtime_reached else 'NOT_REACHED')
print("train/raw_generated_token_count:", raw_status)
if raw_rows:
    print([
        {
            'raw': row['train/raw_generated_token_count'],
            'final': row['train/generated_token_count'],
            'invariant': row['train/raw_generated_token_count'] >= row['train/generated_token_count'],
        }
        for row in raw_rows
    ])
"""
    ),
    markdown("## 20. Canonical reports and final verdict"),
    code(
        """
stage_totals = {'Stage 1': 10, 'Stage 2': 6, 'Stage 3': 2, 'Stage 4 One-sided': 7, 'Stage 4 Credit': 4}
stage_passes = {
    stage: sum(row['status'] == 'PASS' for row in canonical_rows if row['stage'] == stage)
    for stage in stage_totals
}
core_passes = sum(stage_passes.values())
contamination_ok = all(row['status'] == 'PASS' for row in safety_rows)

def final_gate_passes(core_metric_passes, contamination_passed, raw_metric_status, probe_rng_restored, gates):
    required_gates = (
        'GIT_IDENTITY_GATE',
        'CURRENT_GIT_29_METRIC_SCHEMA_GATE',
        'CURRENT_GIT_STAGE34_RUNTIME_ENTRYPOINT_GATE',
        'KAGGLE_DUAL_T4_HARDWARE_IDENTITY_GATE',
        'CURRENT_GIT_RUNTIME_INSTALLER_GATE',
        'KAGGLE_DUAL_T4_HARDWARE_GATE',
        'KAGGLE_DUAL_T4_FORMAL_GATE',
        'AUTHOR_SFT_MODEL_ASSET',
        'GSM8K_DATA_IDENTITY_GATE',
        'LATENT_END_MARKER_GATE',
        'CONFIG_DRY_RUN_GATE',
        'CLEAN_VALIDATION_OUTPUT_GATE',
        'REAL_STAGE123_RUNTIME_GATE',
        'STAGE123_PASSIVE_INSTRUMENTATION_GATE',
        'REAL_STAGE4_CHECKPOINT_PROBE_GATE',
    )
    return (
        core_metric_passes == 29
        and contamination_passed is True
        and raw_metric_status == 'PASS'
        and probe_rng_restored is True
        and all(gates.get(name, {}).get('status') == 'PASS' for name in required_gates)
    )

final_ok = final_gate_passes(core_passes, contamination_ok, raw_status, probe_rng_restore_succeeded, GATES)

artifact_payload = {
    'expected_commit': EXPECTED_COMMIT,
    'gates': GATES,
    'capability': CAPABILITY,
    'blockers': BLOCKERS,
    'metrics': canonical_rows,
    'non_pollution': safety_rows,
    'raw_generated_token_count_status': raw_status,
    'kaggle_29_metric_runtime_gate': 'PASS' if final_ok else 'BLOCKED',
    'kaggle_30_capture_gate': 'PASS' if final_ok else 'BLOCKED',
    '3gpu_final_gate': 'TARGET_RUNTIME_REQUIRED',
}
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
(OUTPUT_ROOT / 'canonical_29_metric_results.json').write_text(json.dumps(artifact_payload, indent=2, default=str) + "\\n", encoding="utf-8")
(OUTPUT_ROOT / 'canonical_29_metric_results.csv').write_text(
    'metric,stage,family,value,available,source_table,runtime_reached,invariant_check,status,unavailable_reason\\n'
    + '\\n'.join(
        ','.join(json.dumps(row[column], default=str) for column in canonical_columns)
        for row in canonical_rows
    ) + '\\n',
    encoding='utf-8',
)
(OUTPUT_ROOT / 'canonical_non_pollution_results.json').write_text(json.dumps(safety_rows, indent=2, default=str) + "\\n", encoding="utf-8")

print("=" * 60)
print("LATENT-GRPO KAGGLE 2xT4 30-METRIC CAPTURE VALIDATION")
print("=" * 60)
print("\\nGit commit:")
print(EXPECTED_COMMIT)
print()
for stage in ('Stage 1', 'Stage 2', 'Stage 3', 'Stage 4 One-sided', 'Stage 4 Credit'):
    print(f"{stage}: {stage_passes[stage]} / {stage_totals[stage]}")
print(f"\\nCORE METRICS: {core_passes} / 29")
print("\\nraw_generated_token_count:")
print(raw_status)
print("\\nTraining contamination:")
print("PASS" if contamination_ok else "BLOCKED")
print("\\nKAGGLE_29_METRIC_RUNTIME_GATE:")
print("PASS" if final_ok else "BLOCKED")
print("\\nKAGGLE_30_CAPTURE_GATE:")
print("PASS" if final_ok else "BLOCKED")
print("\\n3GPU_FINAL_GATE:")
print("TARGET_RUNTIME_REQUIRED")
print("=" * 60)
print("Artifacts:")
for path in sorted(OUTPUT_ROOT.glob('canonical_*')):
    print(path)
"""
    ),
    markdown(
        """
## 21. Result boundary

Only `CORE METRICS: 29 / 29` together with a complete non-pollution pass permits `KAGGLE_29_METRIC_RUNTIME_GATE: PASS`. Even then, this validates the real Kaggle 2xT4 engineering runtime only. Three-worker aggregation, target BF16, target topology, 3-GPU/FSDP checkpoint probing, target CUDA RNG preservation, memory, and throughput remain target-runtime work.
"""
    ),
]

notebook = nbformat.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
)
nbformat.write(notebook, OUTPUT)
print(OUTPUT)
