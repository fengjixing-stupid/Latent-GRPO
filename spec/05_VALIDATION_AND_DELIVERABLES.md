# Latent-GRPO 验收标准与最终交付


## 0. 路径约定

Codex 必须从项目根目录工作。固定路径为：

```text
./Latent-GRPO       # 作者原始仓库
./spec              # 全部任务规范 Markdown
```

引用任何规范文件时都使用 `./spec/<filename>.md`，不得到 `./Latent-GRPO/spec` 或其他目录中查找。

---

## 1. 验收原则

只有同时满足以下条件才能声明完成：

1. 代码存在；
2. 关键测试真实执行；
3. 单卡 smoke 成功；
4. 三卡初始化和最小训练成功；
5. 输出 validator 成功；
6. `./spec/target_variables.md` 有完整追踪；
7. 任何未实现字段均明确 unavailable_with_reason；
8. Support/probe 不改变训练状态；
9. 用户有准确运行命令；
10. 最终报告不夸大复现级别。

---

## 2. 必须交付的代码

至少：

```text
train_latent_grpo.py
latent_grpo_runner/
configs/
scripts/
tests/
patches/                         # 只有需要上游 patch 时
```

核心脚本：

```text
scripts/run_smoke.sh
scripts/run_3gpu_low.sh
scripts/run_3gpu_high_smoke.sh
scripts/validate_outputs.py
```

建议补充：

```text
scripts/inspect_environment.py
scripts/inspect_checkpoint.py
scripts/summarize_run.py
```

---

## 3. 必须交付的文档

```text
README.md
docs/repo_audit.md
docs/implementation_plan.md
docs/decision_log.md
docs/requirements_traceability_matrix.md
docs/operator_runbook.md
docs/upstream_changes.md             # 若修改上游
docs/metrics_schema_reference.md
docs/progress.md
FINAL_IMPLEMENTATION_REPORT.md
work_reports/agent_*.md
```

`operator_runbook.md` 必须给出从 VSCode terminal 开始的准确命令，不使用 Docker。

---

## 4. 静态检查

至少执行并记录实际结果：

```text
python -m compileall ...
pytest ...
ruff/flake8（若项目采用）
mypy/pyright（若项目采用）
```

不强制为了形式引入所有工具，但：

- Python 语法必须通过；
- 核心配置必须可解析；
- import 必须通过；
- 测试必须能独立运行；
- 不得存在硬编码用户绝对路径；
- 不得记录 secret；
- 不得把作者仓库复制进输出目录。

---

## 5. 单元测试矩阵

## 5.1 统计与 mask

必须测试：

```text
mean/std from sum/sum_sq/count
rate numerator/count
empty mask
NaN handling
different masks use different counts
zero-advantage denominator
response/latent mask
trajectory classification
overlong overlap
group count identity
```

## 5.2 generated token count

构造包含：

```text
normal trajectory
EOS
overlong trajectory
retained retry
discarded internal retry
filtered actor-update trajectory
```

验证：

- 只统计最终训练 rollout trajectory；
- overlong 进入 reward/advantage 时不因后续过滤丢失；
- discarded internal retry 不混入；
- worker integer sums 正确全局求和；
- 不用均值×batch size 估计。

## 5.3 Stable trajectory ID

验证：

- repeat 后创建；
- reorder/filter/select 后仍可追踪；
- 同 group 内唯一；
- resume/重复输入规则稳定；
- local batch index 不落盘为 trajectory ID。

## 5.4 Support

验证：

```text
同 trajectory
同 latent position
K 一致
rollout/pre-update 时间点
retention
top1-in-set
position-weighted aggregation
shape mismatch -> unavailable
不静默截断
```

## 5.5 One-sided / FlipGrad

验证：

```text
surrogate margin 公式
negative rate
near-zero threshold
flip trigger
独立 count
p05
empty mask
```

## 5.6 Credit

默认关闭时验证：

```text
schema 存在
available=false
reason=disabled_by_config
不执行 autograd
```

开启实验路径时验证：

```text
u_i 符号
q_i 归一化
top1 share
effective K
Spearman tie
constant vector
zero gradient
alignment definition
不污染 .grad
```

## 5.7 Storage

验证：

```text
JSON atomic replace
Parquet part readable
schema mismatch
temp file recovery
manifest rebuild
duplicate primary key
resume append
partial write
list field
null + availability
validator nonzero exit
```

---

## 6. 集成测试

## 6.1 Dry run

命令示例：

```bash
python train_latent_grpo.py \
  --config configs/smoke.yaml \
  --dry-run \
  --validate-config
```

验证：

- 上游仓库路径；
- 配置；
- 模型/data/cache；
- 环境；
- schema；
- 输出目录；
- 不加载完整训练或做参数更新。

## 6.2 单卡 smoke

示例：

```bash
CUDA_VISIBLE_DEVICES=0 \
python train_latent_grpo.py \
  --config configs/smoke.yaml \
  --max-steps 2
```

验证：

- 至少一次 rollout；
- reward；
- advantage；
- actor update；
- checkpoint；
- Stage 1/2；
- 至少一次 eval；
- 可选一次 Support；
- 可选一次 one-sided probe；
- 输出 validator 通过。

## 6.3 三卡 smoke

示例：

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
torchrun --standalone --nproc_per_node=3 \
  train_latent_grpo.py \
  --config configs/3gpu-low.yaml \
  --max-steps 2
```

验证：

- 三个 rank 初始化；
- rank/设备映射正确；
- 无重复 writer；
- 分布式统计正确；
- 至少一次参数更新；
- checkpoint 可加载；
- 日志主键无重复；
- validator 通过；
- 输出峰值显存和时间。

若目标机器本轮不可用，可以完成模拟/CPU 测试，但最终报告必须明确“三卡未实测”，不得声称完成目标硬件验证。

---

## 7. Resume 测试

过程：

1. 跑 2 step；
2. 保存 checkpoint；
3. 停止；
4. resume 到 4 step；
5. 验证 step、optimizer step、RNG、part 编号和主键；
6. validator；
7. 与连续 4 step 的可比行为进行合理对照。

检查：

```text
is_resume_run
resume_from_step
last_committed_global_step
checkpoint_step
no duplicate parts
no duplicate primary keys
```

---

## 8. Probe 安全测试

checkpoint probe 前后比较：

```text
model parameter hashes
optimizer state hashes
scheduler state
normal parameter .grad
CPU RNG
CUDA RNG per device
Python random
NumPy RNG
global_step
optimizer_step
cumulative_rollout_tokens
```

要求：

- `probe_performs_optimizer_step=false`；
- probe token 不进入训练计数；
- RNG 恢复成功；
- 参数与 optimizer state 不变；
- credit 路径不污染训练 `.grad`；
- 失败时 family unavailable 或 probe 失败，不继续伪造数据。

---

## 9. 性能验收

至少测：

```text
logging disabled
Stage 1/2 enabled
Support enabled at configured step
checkpoint one-sided probe
checkpoint credit probe（若启用）
```

记录：

```text
step time
metrics compute time
metrics write time
support extra time
probe extra time
peak allocated/reserved memory
writer queue peak
part size
row count
```

不设未经实测的绝对性能承诺，但需：

- 解释开销来源；
- 证明没有每 step full-vocabulary 保存；
- 证明 credit 不是每 step 执行；
- 给出可调 flush/support/probe 参数；
- 若开销异常，修复后再标为 verified。

---

## 10. 指标覆盖验收

`requirements_traceability_matrix.md` 必须包含 `./spec/target_variables.md` 的全部目标字段。

生成机器可读报告：

```text
validation/target_variable_coverage.json
```

建议结构：

```json
{
  "schema_version": "...",
  "total_target_fields": 0,
  "implemented_fields": 0,
  "verified_fields": 0,
  "unavailable_with_reason_fields": 0,
  "blocked_fields": 0,
  "missing_fields": []
}
```

验收要求：

```text
missing_fields == []
```

注意：

- credit 默认关闭不等于字段缺失；
- diagnostic 默认关闭不等于字段缺失；
- 字段可以 unavailable_with_reason；
- 不允许完全不定义字段。

---

## 11. 最终报告格式

`FINAL_IMPLEMENTATION_REPORT.md` 至少包含：

```text
1. Executive summary
2. Environment actually tested
3. Upstream repository commit and integration method
4. Files created
5. Files modified
6. Training profiles
7. Commands actually run
8. Test results
9. Single-GPU smoke result
10. Three-GPU result
11. Target-variable coverage
12. Storage layout and example
13. Performance and peak memory
14. Resume result
15. Probe no-mutation result
16. Upstream patches
17. Unavailable or blocked fields
18. Known limitations
19. Exact next commands for the user
```

必须区分：

```text
implemented
tested with synthetic data
tested on single GPU
tested on 3 GPUs
not tested
```

不得把 synthetic test 写成真实训练验证。

---

## 12. 最终完成检查表

- [ ] 作者仓库从 `./Latent-GRPO` 读取。
- [ ] 未要求 Docker。
- [ ] `train_latent_grpo.py` 可执行。
- [ ] 三个 profile 存在。
- [ ] target variables 无漏项。
- [ ] 被禁止字段未出现。
- [ ] JSON/Parquet schema 明确。
- [ ] 原子写盘、resume、去重通过。
- [ ] 单卡 smoke 通过。
- [ ] 三卡最小训练通过，或明确标记未实测。
- [ ] Support 对齐通过。
- [ ] one-sided probe 通过。
- [ ] credit 默认关闭。
- [ ] probe no-mutation/RNG restore 通过。
- [ ] validator 通过。
- [ ] reviewer blocker 全部解决。
- [ ] 最终报告包含真实运行证据。
