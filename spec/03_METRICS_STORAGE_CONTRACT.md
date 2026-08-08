# Latent-GRPO 中间变量存储协议


## 0. 路径约定

Codex 必须从项目根目录工作。固定路径为：

```text
./Latent-GRPO       # 作者原始仓库
./spec              # 全部任务规范 Markdown
```

引用任何规范文件时都使用 `./spec/<filename>.md`，不得到 `./Latent-GRPO/spec` 或其他目录中查找。

---

## 1. 权威契约

必须记录的字段、公式、mask、count、时间语义和禁止项由：

```text
./spec/target_variables.md
```

唯一决定。

本文件只规定如何组织、编码、写入、恢复和验证这些数据。

---

## 2. 存储格式选择

## 2.1 JSON：小型静态或单实例数据

使用 UTF-8 JSON：

```text
run_config.json
platform_config_snapshot.json
schema_manifest.json
run_status.json
support_definition.json
probe_definition.json
eval_dataset_manifest metadata
```

JSON 要求：

- 稳定 key；
- 明确 schema version；
- `NaN` 不使用非标准 JSON token，改为 `null` 并配 availability；
- 原子写入：临时文件 + fsync（可配置）+ rename；
- 不重复保存每行都相同的大段定义。

## 2.2 Parquet dataset：权威动态表

以下动态表使用 append-only 的分片 Parquet dataset：

```text
train_step_metrics
train_group_metrics
eval_question_results
eval_clean_topk
gumbel_diagnostics
support_metrics
support_benchmark_metrics
probe_metrics
probe_benchmark_metrics
```

不要持续重写一个单体 `.parquet` 文件。采用：

```text
<table_name>/
├── _schema.json
├── _SUCCESS_PARTS.json
├── part-000000-<uuid>.parquet
├── part-000001-<uuid>.parquet
└── ...
```

或等价的 partitioned dataset。

建议按以下字段分区，避免过度产生小目录：

```text
profile_name
seed
必要时 checkpoint_step
```

`global_step` 通常放列中，不必每步建目录。

## 2.3 JSONL：仅可选 debug

可以提供：

```text
debug/events.jsonl
```

用途：

- 人工快速检查；
- 最小 smoke；
- 故障诊断。

它不是权威指标源，不得与 Parquet 产生不同语义。默认关闭或限制行数。

## 2.4 禁止的主格式

不得使用 CSV 作为主要存储格式，因为：

- list/top-K 字段编码不稳定；
- null 与类型容易丢失；
- schema/version 难约束；
- resume 和大规模分析不可靠。

---

## 3. 推荐输出目录

```text
<output_root>/
└── <profile_name>/
    └── seed_<seed>/
        ├── run_config.json
        ├── platform_config_snapshot.json
        ├── schema_manifest.json
        ├── run_status.json
        ├── logs/
        │   └── train.log
        ├── checkpoints/
        │   ├── step_00000010/
        │   └── ...
        ├── metrics/
        │   ├── train_step_metrics/
        │   ├── train_group_metrics/
        │   ├── support_definition.json
        │   ├── support_metrics/
        │   ├── support_benchmark_metrics/
        │   ├── probe_definition.json
        │   ├── probe_metrics/
        │   ├── probe_benchmark_metrics/
        │   └── diagnostics/
        │       └── gumbel_diagnostics/
        ├── eval/
        │   ├── eval_dataset_manifest.parquet
        │   └── checkpoint_00000010/
        │       ├── eval_question_results/
        │       └── eval_clean_topk/
        ├── debug/
        └── validation/
            ├── schema_validation.json
            ├── primary_key_validation.json
            └── completeness_validation.json
```

不得按“所有变量一个 JSON 文件”保存，也不得把逐 step、逐 group、逐 trajectory、逐 question、逐 latent position 数据混在同一张表。

---

## 4. Schema 规则

## 4.1 显式类型

建议类型：

```text
step/count: int64
trajectory_id/generation_id/latent_position: int64
boolean: bool
loss/rate/mean/std: float64 或经过论证的 float32
token_id: int32 或 int64
hash/id/string: string
timestamp/wall time: float64 或 timestamp
list token ids: list<int32/int64>
list probabilities: list<float32>
```

K 固定且 Parquet 库可靠支持时，可使用 fixed-size list；否则使用 list 并同时保存 `clean_topk_k` 或 schema 定义。

所有字段在 `_schema.json` 或 `schema_manifest.json` 中包含：

```text
name
logical_type
physical_type
nullable
stage
record_type
unit
definition_version
primary_key_member
availability_family
```

## 4.2 Null、NaN 与 unavailable

规则：

- 不可用值写 null；
- 同行写 family 或 metric availability=false；
- reason 使用稳定、可机器分析的短字符串；
- 详细堆栈写日志，不塞入 Parquet 每行；
- 退化统计可写 null/NaN，但必须有 count 和 reason；
- 不用 0 代表“没有数据”。

推荐 reason：

```text
disabled_by_config
missing_runtime_interface
empty_effective_mask
shape_mismatch
alignment_failed
degenerate_constant_vector
zero_gradient
checkpoint_missing_metadata
runtime_probe_failed
write_failure
unsupported_upstream_version
```

---

## 5. 主键与重复数据

主键至少遵循 `./spec/target_variables.md`。此外：

```text
train_step_metrics:
  profile_name + seed + global_step

train_group_metrics:
  profile_name + seed + global_step + group_id

support_metrics:
  profile_name + seed + global_step + group_id + trajectory_id + trajectory_class

probe_metrics:
  profile_name + seed + checkpoint_step + probe_batch_id
  + trajectory_group + latent_position_group

probe_benchmark_metrics:
  profile_name + seed + checkpoint_step + probe_batch_id

gumbel_diagnostics:
  profile_name + seed + diagnostic_run_id + diagnostic_batch_index
```

`eval_question_results` 与 `eval_clean_topk` 使用 `./spec/target_variables.md` 指定的主键。

写 part 前：

1. 内存批次内检查主键重复；
2. resume 时检查已提交 part 的 step/checkpoint 范围；
3. part manifest 记录 min/max key；
4. validator 做全量或增量重复检查；
5. 检测到冲突时默认停止写入，不静默覆盖。

---

## 6. 原子写入与提交协议

每个 part：

1. 写入同目录临时文件；
2. 关闭 writer；
3. 校验可读、schema 和行数；
4. 原子 rename 为最终 part；
5. 更新成功 part manifest；
6. 仅成功提交后推进 writer checkpoint。

进程崩溃后：

- `.tmp` 不视为有效；
- 已 rename part 视为有效；
- manifest 丢失时可通过扫描 part 重建；
- 不删除无法识别文件，先移动到 quarantine 或报告。

`run_status.json` 状态：

```text
initializing
running
failed
interrupted
completed
```

并记录：

```text
last_committed_global_step
last_committed_optimizer_step
last_checkpoint_step
last_error_type
last_error_message
updated_at
```

---

## 7. 缓冲与写盘

建议：

- rank 0 维护有界事件队列；
- 小批量转为 Arrow RecordBatch；
- 按记录数、字节数或 step 间隔 flush；
- 写盘线程只处理 detached CPU 数据；
- 计算图 Tensor 不进入队列；
- 队列满时按配置阻塞或降级，不能无声丢数据；
- 关键 core metric 不允许 silent drop；
- debug JSONL 可在压力下丢弃，但要计数。

每个动态表保存：

```text
metrics_compute_time
metrics_write_time
```

其中：

- compute time 不含磁盘写盘；
- write time 不混入 train/step_time；
- 计时定义写入 schema manifest。

---

## 8. 分布式统计包

worker 返回的统计包建议统一为：

```python
{
    "sum": ...,
    "sum_sq": ...,
    "count": ...,
    "nan_count": ...,
    "masked_count": ...,
    "min": ...,
    "max": ...,
    "numerator_count": ...
}
```

driver 合并：

```text
global_sum = sum(worker_sum)
global_sum_sq = sum(worker_sum_sq)
global_count = sum(worker_count)
```

然后计算：

```text
mean = global_sum / global_count
variance = max(global_sum_sq / global_count - mean^2, 0)
std = sqrt(variance)
rate = global_numerator_count / global_count
```

需要无偏样本标准差时必须在 definition version 中明确；默认不要混用 population/std 与 sample/std。

分位数 `onesided/delta_p05`：

- checkpoint probe 数据量受限，可将有限、detached 的有效 margin 传到 driver；
- 或采用明确误差界的分位数 sketch；
- 不得把训练全量 margin all-gather；
- 算法写入 definition version。

---

## 9. Top-K 与列表字段

`eval_clean_topk`：

```text
clean_topk_token_ids
clean_topk_probs
clean_topk_k
```

要求：

- 两个列表长度相等；
- 与 `clean_topk_k` 一致；
- token ID 为整数；
- 概率有限且非负；
- 若语义是归一化 top-K 内权重，检查和；若是原分布概率，不强制和为 1；
- 概率语义写 definition；
- 逐 latent position 一行；
- 不写 full vocabulary。

Support 的 top-K 集合只用于低频临时计算，默认不持久化完整两侧集合；只保存规定的 retention 指标和 count。

---

## 10. Definition 与 schema version

至少维护：

```text
metrics_schema_version
record_version
response_length_definition_version
latent_length_definition_version
generated_token_count_definition_version
length_counting_rule_version
group_definition_version
trajectory_classification_version
overlong_definition_version
support_definition_version
probe_definition_version
credit_definition_version
surrogate_alignment_definition_version
evaluation_rule_version
```

变更以下任一内容必须升级相应版本：

- 字段意义；
- mask；
- count；
- 时间点；
- 聚合算法；
- 分位数算法；
- EOS/stop 计数；
- 轨迹分类；
- Top-K 概率空间；
- alignment 符号规则。

只改变注释、日志文字或文件排序不必升级指标 definition。

---

## 11. Resume 规则

resume 时：

1. 读取 checkpoint 中的 step、RNG 和 config hash；
2. 读取各表 part manifest；
3. 验证 schema version；
4. 确认没有超出 checkpoint 的“未来训练记录”，或明确隔离；
5. 新 part 编号继续递增；
6. 不重写旧 part；
7. 已存在相同主键时停止并报告；
8. `is_resume_run=true`；
9. `resume_from_step` 写入动态记录；
10. checkpoint eval/probe 重跑时允许相同 checkpoint_step，但必须使用不同 `evaluation_run_id` 或在启动前明确 replace/quarantine 策略，不能生成重复主键。

若当前 target schema 没有 `evaluation_run_id`，默认同一 run 内不重复执行相同 checkpoint/question/generation；重跑使用新的输出 run 目录。

---

## 12. 输出验证器

必须提供：

```text
scripts/validate_outputs.py
```

至少检查：

- JSON 可解析；
- 所有 Parquet part 可读取；
- schema 一致；
- 必填字段存在；
- 主键无重复；
- step 单调或符合执行语义；
- count 非负；
- rate 在 [0,1] 或 null；
- std 非负或 null；
- availability=false 时值和 reason 合理；
- `correct + non_correct = trajectory_count`；
- overlong 不被错误作为互斥第三类；
- generation_id 完整；
- clean top-K 列表长度一致；
- probe 不计入 cumulative rollout tokens；
- 普通表不出现 checkpoint_step；
- 被明确禁止的字段不存在；
- credit 默认关闭时 schema 与 reason 正确；
- part manifest 与真实文件一致。

验证输出：

```text
validation/schema_validation.json
validation/primary_key_validation.json
validation/completeness_validation.json
```

任何失败返回非零退出码。
