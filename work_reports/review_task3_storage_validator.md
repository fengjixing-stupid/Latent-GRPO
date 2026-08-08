# Task 3 independent review — storage, resume, validator, and coverage

**Review scope:** `spec/03_METRICS_STORAGE_CONTRACT.md`, storage/coverage requirements in
`spec/05_VALIDATION_AND_DELIVERABLES.md`, the supplied implementation, tests, fixture,
and `work_reports/agent_h_storage_validator.md`. This is a read-only review; no source
was changed.

## Verdict

- **Spec verdict: NO-GO.** The required all-target-field coverage/schema contract is not
  met, crash recovery can silently orphan a committed part, and the output validator has
  trivial pass-through paths for invalid output layouts.
- **Quality verdict: NO-GO.** The focused tests are genuine unit tests for the fake
  backend and pass, but they do not establish the advertised production Parquet/resume/
  validation guarantees. The report correctly says PyArrow integration is deferred; that
  deferment prevents acceptance rather than satisfying it.

## Findings

### P0 — coverage declares completeness without comparing the canonical target list

`latent_grpo_runner/validation/target_coverage.py:29-42` parses only the RTM, never
`spec/target_variables.md`; `:62` hard-codes `missing_fields: []`. Therefore deleting a
canonical target field from the RTM would still result in a passing empty missing list.
Further, every RTM row is presently `planned` (397) or `blocked` (23), yet `:48-61` calls
the intersection of three partial schema tables and RTM rows `implemented_fields=93`.
There is no per-field status/evidence in `validation/target_variable_coverage.json` to
support that claim.

More specifically, `_LEGACY_STATUS = {"planned": "unavailable_with_reason"}` at
`target_coverage.py:17` silently converts all 397 merely planned rows to
`unavailable_with_reason`. Planning is neither an unavailable declaration nor its required
stable reason, so this conversion hides unimplemented work behind an acceptance-shaped count.

The actual schema manifest has only `train_step_metrics`, `train_group_metrics`, and
`gumbel_diagnostics` (`metrics/schemas.py:107-114`). The canonical contract also requires
schemas for `support_metrics`, `support_benchmark_metrics`, `eval_question_results`,
`eval_clean_topk`, `probe_metrics`, and `probe_benchmark_metrics` (and their unavailable
semantics if runtime work is deferred). This violates the rule that default-disabled or
unavailable fields must still be defined rather than omitted. The reported **420 / 330**
counts happen to match the RTM's self-count, but are not verified against the authority and
must not be used as an acceptance result.

### P1 — recovery loses a valid renamed part when the manifest is stale

`AppendOnlyPartWriter._load_or_rebuild_manifest()` (`metrics/storage.py:147-160`) rebuilds
only when `_SUCCESS_PARTS.json` is absent. If a process crashes after `os.replace()` at
`:217` but before the manifest update at `:220`, the next writer trusts the old manifest and
does not scan/reconcile the already-renamed part. It consequently does not load its keys.
This conflicts with the contract that renamed parts are valid and that recovery must rebuild
manifest state.

Reproduction using the project's `JsonBackend` test double: commit parts for steps 1 and 2,
restore the pre-step-2 manifest, then construct a new writer. The directory had two
`part-*.parquet` files, while the resumed manifest and loaded key set contained only step 1.
The existing test covers a *missing* manifest, not this required stale-manifest crash window.

### P1 — validator can return success for corrupt/undeclared metrics and does not enforce key contract rules

`validate_output_directory()` (`validation/output_validator.py:101-120`) iterates only
tables declared in `schema_manifest.json`, and silently continues when one is missing
(`:102-104`). It neither discovers unexpected metric table directories nor requires expected
tables/fields. In an isolated temporary output directory with a valid empty `tables` object
and a corrupt `metrics/train_step_metrics/part-*.parquet`, the actual CLI returned **0**.
Thus the checked-in fixture's empty schema is a validation shell, not an exercised output
dataset.

`validate_records()` also accepts all of the following required-invalid cases: a
`checkpoint_step` in `train_step_metrics`; `model_export_path` (a prohibited artifact
reference); `train_core_available=false` with no reason; and top-K string token IDs with a
negative probability. It checks only selected substrings and lengths, not required field
presence/types/nullability, ordinary-table `checkpoint_step`, complete forbidden-field policy,
availability-family reasons, top-K integer/finite/nonnegative values, or the probe cumulative
token invariant. These are explicit checks required by storage contract §12.

### P1 — append writer does not provide durable single-writer Parquet commit semantics

The JSON helper is well structured: sibling temp file, file fsync, atomic replace, and
directory fsync (`metrics/storage.py:42-63`). The Parquet path, however, has no fsync of the
temporary part after backend close and before rename (`:198-217`). `fsync=True` therefore
does not durably cover the authoritative part data.

`PyArrowBackend.write()` ignores its supplied manifest schema (`metrics/storage.py:78-84`) and
uses `pa.Table.from_pylist()` type inference; the backend reports only Arrow field names on
read (`:86-88`). The later check compares names, not manifest physical/logical types
(`:207-212`). A row value can therefore choose a physical Arrow type inconsistent with the
declared manifest and still pass writer readback. This is not the required schema-driven
Parquet lifecycle.

`writer_rank != 0` (`:123-124`) is only a caller-supplied guard, not mutual exclusion. Two
rank-0 processes can independently read the same manifest, choose the same next number, and
overwrite each other's in-memory manifest updates; duplicate-key protection is likewise only
per writer instance. No lock/lease or driver-integrated ownership path exists. This falls
short of the single-writer and resume duplicate guarantees.

### P1 — PyArrow lifecycle is deliberately deferred, so its required behavior is unverified

The lazy import is correct (`metrics/storage.py:69-76`), and `pyarrow` is absent on this Mac.
However, all seven storage tests use `JsonBackend`, which writes JSON bytes with a `.parquet`
name (`tests/unit/test_storage.py:7-14`). No test has performed actual Arrow schema/type/list/
null write-readback, real Parquet corruption handling, or manifest/resume behavior against
Parquet. Agent H accurately labels this target-machine work deferred; it should remain an
acceptance blocker, not be summarized as “Parquet part readable” having been verified.

### P2 — resume/quarantine helpers are not connected to a recovery workflow

`checkpointing.validate_resume()` only compares a caller-provided map of maximum steps
(`checkpointing.py:36-48`); it does not inspect manifests/parts, quarantine future data, or
advance part state. `quarantine_future_part()` is a standalone move helper (`:50-59`) with
no production call site (repository search finds only its unit test). The unit test proves a
move operation, not the specified checkpoint → manifest scan → future isolation → append
resume flow.

### P2 — fixture labeling is appropriately non-GPU, but it is not a sample training run

Positive: `tests/fixtures/sample_run/platform_config_snapshot.json` explicitly has
`target_gpu_environment_available=false`, and no GPU completion is claimed. Negative: it has
no declared dynamic tables or metrics parts, so the validator CLI test demonstrates only
JSON/report creation, not a valid sample-run storage layout.

## Evidence executed (project-default `python3`; no dependencies installed)

- `python3 -m unittest tests.unit.test_storage tests.unit.test_resume tests.unit.test_output_validator tests.unit.test_target_coverage -v` → **18/18 passed**.
- `python3 -m unittest discover -s tests/unit -v` → **67/67 passed** in this environment.
- `python3 -m compileall -q latent_grpo_runner scripts tests` → **0**.
- `python3 scripts/validate_outputs.py --input tests/fixtures/sample_run` → **0**; the fixture contains no dynamic dataset.
- `pyarrow_present=False`; production Parquet behavior was not runnable without installing it.
- Isolated reproductions described in P1 confirmed both stale-manifest data loss and a corrupt undeclared table returning validator exit **0**.

## Required disposition

Do not claim storage/validator/coverage acceptance. At minimum, fix the P0 coverage/schema
contract and P1 recovery/validator/single-writer failures, add regression tests for each
reproduction above, then run real PyArrow lifecycle tests on the target environment before a
GO verdict.

---

## Round 1 scoped re-review

**Task 3 scope verdict: still NO-GO for final acceptance; GO to the isolated PyArrow
integration stage.** The revised lazy backend has the right import boundary and now converts
the manifest into an explicit Arrow schema, but production Parquet behavior has not yet run.
Remaining validation/schema/resume gaps also prevent a storage acceptance claim.

| Original finding / requirement | Disposition | Evidence and current assessment |
|---|---|---|
| Coverage falsely counted declared fields as implemented | **ADDRESSED** | `implemented_fields` is now `0`; `schema_declared_fields=93` is separately reported, and `coverage_is_acceptance_ready=false`. |
| `planned` was mapped to `unavailable_with_reason` | **ADDRESSED** | It now maps to `target_machine_test_deferred`; current report has `unavailable_with_reason_fields=0` and `target_machine_test_deferred_fields=397`. This no longer asserts a stable unavailable reason that does not exist. |
| `missing_fields=[]` must be proven from the canonical target list and every unavailable field must have a schema | **NOT ADDRESSED** | `target_coverage.py:28-64` still reads only the RTM and still returns `missing_fields: []` unconditionally. `schemas.py:108-125` adds six deferred table names but gives each `fields: []`; only 93/420 table-qualified target fields are declared. Thus 327 fields still lack the required schema/type/availability definition. The honest `acceptance_ready=false` label mitigates reporting risk but does not meet the contract. |
| Stale manifest after rename loses a committed part/key | **ADDRESSED** | Writer now scans all renamed parts even with a manifest (`storage.py:261-283`), rewrites the manifest, and its regression test restores a stale manifest then verifies two parts and PK rejection (`test_storage.py:116-128`). |
| Unreadable renamed/orphan part must be preserved rather than silently used/deleted | **ADDRESSED** | Recovery moves unreadable parts to `quarantine/`; regression `test_bad_orphan_part_is_quarantined_during_reconciliation` passes. |
| Empty schema / undeclared dynamic table lets validator pass | **ADDRESSED** | Validator requires four static files plus all nine dynamic table declarations (`output_validator.py:124-150`) and detects undeclared directories. I independently confirmed an empty `tables` manifest exits 1. The checked-in fixture exits 0 only because every required table is explicitly declared `target_machine_test_deferred` with a reason; it makes no GPU or Parquet-executed claim. |
| Validator checks forbidden fields, ordinary-table checkpoint_step, family availability reason, and basic top-K validity | **ADDRESSED** | Added checks at `output_validator.py:36-88` and direct regression coverage. |
| Validator validates every declared required field/type/nullability and each Parquet part schema | **NOT ADDRESSED** | `validate_output_directory()` reads parts but never compares the returned Arrow schema to the declared table schema, and `validate_records()` checks only PK/selected semantic rules. Independent call using the 98-field `train_step_metrics` schema with a row containing only its three PK fields returned `[]` (95 declared fields missing). This is still a §12 schema/required-field validation gap. Probe cumulative-rollout-token exclusion is likewise not checked. |
| JSON atomic durability and Parquet temp/rename durability | **ADDRESSED (code review; fake-backend exercised)** | JSON retains file+directory fsync. Append now fsyncs the closed temp part before `os.replace()` and fsyncs the table directory after rename (`storage.py:323-338`); subsequent JSON manifest/checkpoint writes fsync their own file+directory. Unit tests exercised this path with a real temporary filesystem but do not instrument the fsync calls. |
| Schema-driven physical types rather than Arrow inference | **ADDRESSED (unit-level)** | `PyArrowBackend.write()` now validates rows and calls `pa.Table.from_pylist(..., schema=_arrow_schema(...))`; readback returns physical type and nullability, which `_schema_matches()` compares (`storage.py:78-91`, `:119-204`). Fake-backend regressions cover mismatch, missing/extra column, and wrong type. |
| Actual PyArrow write/read/list/null/corruption integration | **NOT ADDRESSED — intentionally deferred** | `pyarrow` remains absent and no dependency was installed. This must be completed in the isolated venv/target environment. |
| Single writer across processes/rank-0 races | **NOT ADDRESSED** | The only authority control remains caller-provided `writer_rank != 0` (`storage.py:237-238`); no lock/lease/driver ownership integration protects two rank-0 processes from racing manifest updates or duplicate keys. |
| Sidecar → manifest scan → future-part quarantine → resumed append workflow | **NOT ADDRESSED** | `checkpointing.py` remains standalone helpers, with no non-test caller. `validate_resume()` still consumes a caller-provided maximum-step map and `quarantine_future_part()` is not connected to recovery. |

### PyArrow readiness decision

**Eligible for the next isolated PyArrow integration test, not eligible for an acceptance
claim.** `pyarrow` is imported only inside `PyArrowBackend._modules()`, so import/config/unit
workflows remain dependency-free. An independent direct call to `PyArrowBackend.write()` on
this Mac raised the documented `ParquetBackendUnavailable` message; it did not fail at module
import. The explicit schema mapping supports the current declared scalar/list types.

The integration gate must use actual PyArrow to cover write → close → fsync → readback, list
and null fields, physical/nullability mismatch, corrupt temporary/final parts, stale manifest
rebuild, and validator behavior on a real dataset. It must also add validator field-schema
comparison before declaring storage acceptance.

### Re-review commands actually run

- `python3 -m unittest discover -s tests/unit -v` → **75/75 passed**.
- `python3 scripts/validate_outputs.py --input tests/fixtures/sample_run` → **0**; fixture is explicitly all-table deferred and GPU-negative.
- `python3 -m compileall -q latent_grpo_runner scripts tests` → **0**.
- Direct no-PyArrow backend call → expected stable `ParquetBackendUnavailable`.
- Direct empty-manifest validator reproduction → **exit 1** with a required-dynamic-table error.
- Direct schema-completeness reproduction → a 3-column row against the 98-field step schema produced **no errors**, confirming the remaining validator finding.

---

## Round 2 scoped re-review — remaining non-PyArrow blockers

**Task 3 scope verdict: FAIL.** Treat real PyArrow round-trip as **Task 6 deferred** for
this review; it is neither a Round 2 failure nor evidence of a pass. The non-PyArrow storage
surface improved materially, but the required canonical coverage/schema contract, probe-token
validator invariant, and end-to-end sidecar resume wiring remain incomplete.

| Item | Verdict | Evidence |
|---|---|---|
| Validator enforces manifest columns, nullability, types, and part schema | **PASS** | `validate_records()` now loops through `_field_specs()`, rejects missing/unexpected fields and null/type violations (`output_validator.py:33-46`); read parts are compared with the table manifest (`:179-186`). The new unit regression passes. My former 3-PK-fields-versus-98-field-schema reproduction now returns **95** errors. |
| Empty/undeclared outputs and explicitly deferred sample fixture | **PASS** | The full unit suite passes and `python3 scripts/validate_outputs.py --input tests/fixtures/sample_run` exits 0. This is legitimate for the fixture because all nine table declarations are explicitly deferred with a reason; the fixture continues to state that target GPU is unavailable. |
| Writer single-writer authority | **PASS, local filesystem scope** | `AppendOnlyPartWriter` now uses nonblocking `fcntl.flock` and an in-process held-path guard (`storage.py:249-280`), with `close()`/context-manager release and a regression test. This is appropriate for the specified Mac/Linux local output directory. It is not a distributed-filesystem lease, which the current contract does not require. |
| Resume writer quarantines future parts before accepting new parts | **PASS, writer API scope** | `resume_checkpoint_step` scans actual committed records and moves a future part to `quarantine/future-step-*` (`storage.py:299-330`). The regression creates step 1/2, resumes at 1, and retains one committed part. Stale-manifest rebuild and cross-part PK checks remain in place. |
| Atomic JSON / part fsync / post-rename directory fsync | **PASS (code + fake-backend coverage)** | The previously reviewed fsync sequence remains present. No regression was observed in 78 unit tests. Real Parquet filesystem lifecycle remains Task 6 deferred. |
| `planned` claimed as unavailable or implemented | **PASS** | Coverage reports `implemented_fields=0`, and maps legacy planned entries to the explicit `target_machine_test_deferred` state rather than `unavailable_with_reason`. |
| Canonical target-variable coverage and full unavailable schema definitions | **FAIL** | `target_coverage.py:59-74` merely locates `target_variables.md` and checks for the literal 29-core-metrics summary; it never derives or compares the canonical field inventory, and still returns `missing_fields: []` unconditionally. The six deferred tables still have `fields: []` (`schemas.py:108-125`), so only 93 of the 420 RTM-qualified targets have schema declarations. Explicit deferral does not satisfy the requirement to define unavailable fields with schema/type/availability semantics. |
| Probe records do not enter cumulative rollout tokens | **FAIL** | No validator rule tests or rejects a probe row that changes/contains `cumulative_rollout_tokens`; search finds no corresponding invariant in `output_validator.py`. Storage contract §12 explicitly requires this check. |
| Checkpoint sidecar → writer resume integration | **FAIL** | The writer API can quarantine when its caller supplies `resume_checkpoint_step`, but `checkpointing.py` is unchanged: it does not instantiate/configure writers, scan manifests, or invoke the quarantine flow. Repository search finds no non-test call site for the checkpoint helpers or writer. Thus the required application-level resume protocol is not wired. |
| Real PyArrow lifecycle | **Task 6 deferred** | Not evaluated as a Round 2 blocker. It still needs the isolated venv tests previously listed. |

### Round 2 commands actually run

- `python3 -m unittest discover -s tests/unit -v` → **78/78 passed**.
- `python3 scripts/validate_outputs.py --input tests/fixtures/sample_run` → **0**.
- `python3 -m compileall -q latent_grpo_runner scripts tests` → **0**.

The tests substantiate the listed PASS items, but they do not remove the three FAIL items;
therefore Task 3 cannot be signed off yet.

---

## Round 3 scoped re-review — inventory, deferred schemas, token invariant, and sidecar factory

**Task 3 scope verdict: FAIL.** The Round 3 storage/validator mechanisms below are sound at
their tested scope. Two **load-bearing** schema/coverage defects remain; the real-PyArrow
lifecycle is a separate, clearly deferred **Task 6** item and is not counted as this verdict's
failure.

| Round 3 item | Verdict | Evidence |
|---|---|---|
| Inventory source caveat is visible and avoids false implemented/unavailable claims | **PASS, transparency only** | The report now contains 420 RTM table-qualified inventory entries, separately reports `implemented_fields=0`, and exposes RTM and canonical-source counts. It continues to label itself not acceptance-ready. `planned` remains mapped to explicitly deferred, not unavailable. |
| Inventory is an acceptance-grade comparison to the canonical target source | **FAIL — load-bearing** | `target_coverage.py:59-77` only looks for the target document, searches for a literal 29-core-metrics summary, and derives 66 unqualified pipe-table names; it neither parses canonical field membership nor computes `missing_fields`, which remains a hard-coded empty list. The stated caveat is honest, but cannot satisfy the required `missing_fields == []` proof. |
| Six deferred dynamic persistent schemas | **PASS** | `_rtm_fields_by_table()` populates deferred-table fields from the RTM (`schemas.py:27-46,148-149`). Independent reconciliation found **134 expected / 134 declared / 0 missing** across `eval_question_results`, `eval_clean_topk`, support, and probe tables. They retain explicit target-machine-deferred reasons. |
| Complete dynamic table schema coverage | **FAIL — load-bearing** | Four RTM-required persistent fields in the already-declared `gumbel_diagnostics` table remain absent: `gumbel_diagnostics_enabled`, `gumbel_compute_time_seconds`, `record_available`, and `record_unavailable_reason`. The separate `eval_dataset_manifest.parquet` contract also has no schema-table definition. These omissions prevent the asserted all-target schema contract even though deferred-table coverage is now fixed. |
| Cumulative rollout token invariant and probe/eval exclusion | **PASS** | Validator now requires step-to-step cumulative continuity using `train/generated_token_count` (`output_validator.py:88-95`) and rejects a training counter on probe/eval rows (`:96-97`). The added regression supplies an invalid 3→10 cumulative sequence and a probe counter; both are rejected. |
| Sidecar writer factory | **PASS, module integration scope** | `resume_metric_writers_from_sidecar()` validates the sidecar, requires manifest entries, constructs writers with `resume_checkpoint_step`, and closes partial construction on failure (`checkpointing.py:62-84`). Its regression writes steps 1/2 then resumes from sidecar step 1 and observes only the valid part. |
| Train/runtime invokes the new factory in a real resume | **Clearly deferred integration evidence** | The new factory is test-covered but has no observed non-test caller in this codebase. This does not invalidate its module contract; its actual upstream training integration belongs with the runtime/resume integration work. |
| Real PyArrow lifecycle | **Task 6 deferred** | No scope change: all actual Arrow/Parquet round-trip, corruption, and validator integration evidence must come from the isolated venv task. |

### Round 3 commands actually run

- `python3 -m unittest discover -s tests/unit -v` → **80/80 passed**.
- `python3 scripts/validate_outputs.py --input tests/fixtures/sample_run` → **0**.
- `python3 -m compileall -q latent_grpo_runner scripts tests` → **0**.
- Deferred-schema reconciliation → **134 expected / 134 declared / 0 missing**.

To change the Task 3 verdict to PASS, make the coverage comparison authoritative rather than
informational and add the four missing Gumbel fields (plus an explicit schema for the eval
dataset manifest). The Task 6 PyArrow test remains a separate acceptance dependency.

---

## Round 4 final scoped re-review — authoritative coverage and Gumbel/eval schemas

**Task 3 scope verdict: FAIL on one load-bearing delivery artifact; code-level findings are
otherwise PASS.** Real PyArrow lifecycle remains **Task 6 deferred**, outside this verdict.
Regenerating the checked-in coverage artifact from the verified current function would remove
the remaining Task 3 failure.

| Finding | Verdict | Evidence |
|---|---|---|
| Unique-name authoritative coverage cross-check | **PASS** | `extract_target_fields()` now conservatively reads explicit declarations and quarantines ambiguous tokens; `build_target_coverage()` calculates `missing_from_rtm` and `extra_in_rtm` from real sets (`target_coverage.py:53-191`). The synthetic regression proves a spec-only field becomes `missing_fields`. Current live output is 325 spec-extracted names, 330 RTM names, `missing_fields=[]`, and no ambiguous tokens. The unavoidable lack of table qualification in `target_variables.md` is correctly documented as a caveat rather than guessed. |
| Gumbel persistent schema omissions | **PASS** | `gumbel_diagnostics` now includes `gumbel_diagnostics_enabled`, `gumbel_compute_time_seconds`, `record_available`, and `record_unavailable_reason` with the RTM-matched types/nullability (`schemas.py:124-133`). Schema regression checks all four. |
| Eval dataset manifest schema | **PASS** | `eval_dataset_manifest.parquet` now has its required composite primary key and six non-null string fields (`schemas.py:150-156`), all verified against RTM by the added regression. |
| Delivered `validation/target_variable_coverage.json` reflects current coverage implementation | **FAIL — load-bearing artifact freshness** | The checked-in JSON is stale: it reports old `coverage_mode="rtm_table_qualified"`, `schema_declared_fields=227`, and lacks `spec_extracted_fields`, `missing_from_rtm`, `extra_in_rtm`, and ambiguity evidence. The current function returns `coverage_mode="rtm_table_qualified_with_spec_unique_crosscheck"` and `schema_declared_fields=237`. There is no non-test call site that writes the current function result to the required deliverable. Do not claim the file as the Round 4 coverage result until regenerated. |
| Actual PyArrow/Parquet round-trip | **Task 6 deferred** | Explicitly excluded from this Round 4 scope verdict. |

### Final Round 4 evidence

- `python3 -m unittest discover -s tests/unit -v` → **83/83 passed**.
- `python3 scripts/validate_outputs.py --input tests/fixtures/sample_run` → **0**.
- `python3 -m compileall -q latent_grpo_runner scripts tests` → **0**.
- Direct `build_target_coverage()` execution → 420 table-qualified RTM records, 330 RTM unique names, 325 extracted spec unique names, and `missing_fields=[]`; the five RTM-only entries are template/aggregate names and are exposed in `extra_in_rtm`, not hidden.

Once the coverage JSON is regenerated from `build_target_coverage()` and committed as the
deliverable, Task 3's non-PyArrow scoped verdict can change to **PASS**. Task 6 remains
required for real Parquet acceptance.

---

## Final artifact freshness verification

**Task 3 non-PyArrow final verdict: PASS.** `validation/target_variable_coverage.json` was
regenerated and I compared its parsed JSON object to a fresh
`build_target_coverage('docs/requirements_traceability_matrix.md')` result: **exact equality
is true**. It now carries the cross-check mode, full inventory, source sets, ambiguity data,
and current `schema_declared_fields=237`.

Current delivered coverage: 420 RTM-qualified records, 330 RTM unique names, 325 explicitly
extracted spec names, `missing_fields=[]`, `missing_from_rtm=[]`, and no ambiguous tokens.
The five disclosed RTM-only entries are templates/aggregate metadata, not concealed misses.
Real PyArrow/Parquet lifecycle validation remains the separate Task 6 deferred acceptance
dependency.
