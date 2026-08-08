# Agent H — storage, resume, output validation, and coverage

## Scope

- Added Mac-safe storage and validation code only under the project root; no files in `./Latent-GRPO` were changed and no dependencies were installed.
- Read the supplied Mac/target-machine task attachment, `spec/03_METRICS_STORAGE_CONTRACT.md`, the storage/coverage portions of `spec/05_VALIDATION_AND_DELIVERABLES.md`, the current metrics schema, and the RTM.

## RED → GREEN evidence

1. **RED:** `python -m unittest tests.unit.test_storage tests.unit.test_resume tests.unit.test_output_validator tests.unit.test_target_coverage -v` initially ran 12 tests; every test failed with the expected missing-module error before implementation.
2. **GREEN:** the focused storage/resume/validator/coverage plus schema suite was rerun after implementation and now contains 22 tests, all passing. It covers JSON atomic replacement; fake-backend list/null round trips and part close/readback/rename/manifest/checkpoint ordering; in-batch and cross-part PK rejection; temp-file recovery; manifest rebuild; schema mismatch; rank-0-only writer; sidecar compatibility; future-part quarantine; validator nonzero conditions and monotonic steps; CLI fixture; RTM counts/status validation; and rejection of full component-log-probability tensors.
3. A later RED/GREEN cycle added explicit readback schema rejection and non-destructive future-part quarantine.
4. `python scripts/validate_outputs.py --input tests/fixtures/sample_run` exits 0 and writes the three validation JSON reports. The fixture explicitly says `target_gpu_environment_available=false`; it makes no GPU claim.
5. `python -m compileall -q latent_grpo_runner scripts tests` exits 0.

## PyArrow deferred status

- PyArrow is absent on this Mac and was not installed, as required. It is imported only inside `PyArrowBackend._modules()`.
- The verified stable deferred error is: `PyArrow is required for Parquet metrics storage; install pyarrow to write or validate parts`.
- Atomic/PK/recovery behavior is unit-tested through an injected filesystem fake backend. Actual PyArrow Parquet list/null/readback integration remains `target_machine_test_deferred` pending the isolated development/target environment installation requested by the lead.

## Coverage report

- `validation/target_variable_coverage.json` is a definition-template report derived from the RTM: 420 table-qualified fields, 330 unique names, and `missing_fields=[]`.
- `implemented_fields=93` is the RTM intersection with the current declared `schema_manifest()` tables; `verified_fields=0` because no target-runtime evidence exists. Legacy RTM `planned` entries are normalized only in this derived report to `unavailable_with_reason`; unknown status labels are rejected.

## Whole-suite note

- `python -m unittest discover -s tests/unit -v` was executed. The new storage/validation tests passed, as did existing metrics/environment tests. The suite has 12 unrelated Task 1/launcher failures because `yaml` is not installed (`ConfigError: PyYAML is required to parse runner profiles`). This agent did not install dependencies or change Task 1 code.

## Review round 1 (storage-validator NO-GO fixes)

- **RED:** added review regressions and ran `python3 -m unittest tests.unit.test_storage tests.unit.test_output_validator tests.unit.test_target_coverage -v`; eight expected failures demonstrated stale manifest orphan loss, no bad-part quarantine, schema-contract gaps, validator pass-through, and the invalid `planned` availability mapping.
- **GREEN:** recovery now reconciles every readable renamed `part-*.parquet` with either a present or stale/missing manifest, rebuilds PK state, and moves unreadable parts into `quarantine/` rather than deleting them. Part data is fsynced after the backend has closed it and before rename; the table directory is fsynced after rename.
- `PyArrowBackend` remains lazy, but now builds an explicit Arrow schema from `name`/`physical_type`/`nullable`, rejects missing/extra/type-invalid row columns, and readback compares names, physical types, and nullability. The same contract is exercised by a strict injected fake backend; PyArrow execution remains deferred.
- The validator now requires four static JSON files and all nine dynamic table declarations, rejects undeclared table directories and non-deferred absent dynamic tables/manifests, and checks family reasons, checkpoint-step placement, forbidden artifact names, and top-K token/probability validity. The sample run declares every absent dynamic table explicitly as `target_machine_test_deferred`; it is no longer an empty-schema bypass.
- RTM legacy `planned` is normalized only to `target_machine_test_deferred`, never to unavailable. The coverage report records `implemented_fields=0`, `schema_declared_fields=93`, and `verified_fields=0`; its `coverage_is_acceptance_ready=false` makes clear it is not an evidence claim for unimplemented runtime metrics.
- Fresh final evidence: `python3 -m unittest discover -s tests/unit -v` → **75/75 passed**; `python3 scripts/validate_outputs.py --input tests/fixtures/sample_run` → **0**; `python3 -m compileall -q latent_grpo_runner scripts tests` → **0**.

## Review round 2

- **RED:** before implementation, the new coverage-inventory, writer-lock, resume-isolation, and manifest-field-contract tests failed because the inventory/lock/close/resume path and validator field checks did not exist.
- **GREEN:** `build_target_coverage()` now emits all 420 RTM table-qualified records as an inventory with per-record `declared`, `implemented`, and `test_status`; it records both the RTM count (420) and the explicit target-variable core-metric count (29). It continues to report `implemented_fields=0`, `verified_fields=0`, and `coverage_is_acceptance_ready=false`.
- The append writer now has a held-lifecycle POSIX `fcntl.flock` plus in-process guard, stable second-writer failure, `close()`, and context-manager release. Resume initialization accepts a checkpoint step and quarantines full future parts before PK state is reconstructed.
- The output validator now applies declared manifest field names, nullability, and physical-type requirements to rows, rejects extra columns, and compares each part's readback schema with the table manifest. These checks are covered through the injected fake backend; no PyArrow roundtrip claim is made.
- Fresh final evidence: `python3 -m unittest discover -s tests/unit -q` → **78/78 passed**; fixture validator → **0**; compileall → **0**. Real PyArrow roundtrip remains explicitly deferred to Task 6.

## Review round 3

- **RED:** inventory source accounting, train-token continuity, probe/eval token exclusion, and sidecar-to-writer factory tests failed before their APIs/checks existed.
- **GREEN:** the metrics package now contains `target_inventory.json`; runtime coverage emits the complete 420-record RTM table-qualified inventory with per-record declaration/implementation/test status, calculated set differences, and cross-source counts from `target_variables.md` (29 core metrics; 66 explicitly tabulated canonical names). `missing_fields` is derived from the canonical RTM inventory set, not a literal.
- Deferred persistent eval/support/probe/benchmark schemas now obtain their field/type declarations from the RTM table-qualified source. Memory-only rows remain excluded from persistence. The current declared persistent intersection is 227 fields; runtime implementation and verification remain 0.
- Validator checks cumulative rollout-token continuity across train rows and rejects cumulative training tokens in eval/probe rows. `resume_metric_writers_from_sidecar()` now opens real append writers from a sidecar, passes its checkpoint step into recovery, validates requested table manifests, and lets writer recovery quarantine future parts before rebuilding valid PK state; this is integration-tested using real writers and the fake backend.
- Fresh final evidence: `python3 -m unittest discover -s tests/unit -q` → **80/80 passed**; fixture validator → **0**; compileall → **0**. Real PyArrow roundtrip is explicitly deferred to Task 6.

## Review round 4 — canonical unique-field cross-check and schema completion

- Scope remained limited to the two Round 3 load-bearing findings. No dependency was installed and no PyArrow execution is claimed.
- **RED:** two coverage tests failed with the expected missing `extract_target_fields` API and missing `spec_extracted_fields` report key. The schema contract test failed with `KeyError: 'gumbel_diagnostics_enabled'` before the schema change.
- **GREEN / authoritative extraction:** `extract_target_fields()` reads explicit field-name Markdown tables, fenced identifier/declaration lists, and explicitly labelled inline field declarations. It excludes prohibition blocks, value-enumeration blocks, formula right-hand sides and constants, and generic `<...>` templates. Identifier-shaped content in explicitly example-only contexts is reported through `ambiguous_tokens` instead of being promoted.
- `build_target_coverage()` now performs the real unique-name set comparison and reports `spec_extracted_fields`, `rtm_unique_fields`, `missing_from_rtm`, `extra_in_rtm`, and `ambiguous_tokens`; `missing_fields` is the calculated `missing_from_rtm` list. Table-qualified inventory remains the RTM's responsibility and is explicitly named by `coverage_mode=rtm_table_qualified_with_spec_unique_crosscheck`.
- Current authority comparison: **325** extracted spec unique fields, **330** RTM unique fields, **0** missing from RTM, **5** extra in RTM, and **0** ambiguous. The five extras are the four deliberately excluded generic templates (`<family>_*`, `<metric_name>__*`) plus RTM-only `numerator_count`; they remain visible rather than being forced into the spec set.
- **Schema completion:** `gumbel_diagnostics` now includes `gumbel_diagnostics_enabled: bool non-null`, `gumbel_compute_time_seconds: float64 nullable`, `record_available: bool non-null`, and `record_unavailable_reason: string nullable`. `eval_dataset_manifest.parquet` is declared with exactly six non-null string fields and primary key `(eval_dataset_name, eval_dataset_version, question_id)`. The regression reads RTM field/type rows and cross-checks both additions.
- Fresh evidence: focused coverage/schema tests → **11/11 passed**; `python3 -m unittest discover -s tests/unit -v` → **83/83 passed**; `python3 scripts/validate_outputs.py --input tests/fixtures/sample_run` → **0**; `python3 -m compileall -q latent_grpo_runner scripts tests` → **0**.
