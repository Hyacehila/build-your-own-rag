# 运行、评分与交付

所有命令在项目根目录执行，使用 `.venv/Scripts/python.exe -B -X utf8`。环境采用 Python 3.11、`uv.lock`；本机保留 OCR 可选依赖。环境检查：`uv sync --locked --check --offline --extra ocr`；安装使用 `uv sync --locked --extra ocr --link-mode copy`。

## 配置与准备

五组名称统一为 A、B、C、pre-D、D。唯一生产配置 `config.toml`：RRF20、两路各50；A 最终选择10个入口，B 最多选择20个原始入口，C 保留 B 的原始入口后增加最多5个摘要入口。B/C六阶段25k/8图，受控原页最后最多4k，不预留5k跨小节额度。`config.offline.toml`只用于虚构测试。

容量来自 `config/gateway-capacity.json` 和原响应证明：131072上下文，16384输出预留，文本/图像代理计数乘1.2；不是准确计费。原文及索引只读复用，不因代码整理重建。

```powershell
.\.venv\Scripts\python.exe -B -X utf8 tools/experiment_setup.py --check --scope full
# 仅在明确取得本次全量授权后使用 --full-run-authorized。
.\.venv\Scripts\python.exe -B -X utf8 tools/experiment_setup.py --output .local/experiments/NEW-RUN --run-id NEW-RUN --scope full --arms A B C --full-run-authorized --reuse-material .local/reused/d-full296-20261008
.\.venv\Scripts\python.exe -B -X utf8 tools/contract_experiment.py --plan .local/experiments/NEW-RUN/plan.json --check
```

准备器创建独立 runtime：零旧答案、零 judge 缓存、零 API 账本；可复用源库身份匹配的 embedding。计划绑定配置、模型、容量、原题/标签、源库、索引和API起始边界。正式复用材料以 `plan.reuse_material` 的路径和来源批次 绑定，复用记录保留原 run_id。任何修改应在启动前完成，运行中不改冻结代码或配置。

## 串行入口与五个独立入口

```powershell
.\.venv\Scripts\python.exe -B -X utf8 tools/contract_experiment.py --plan PLAN.json
.\.venv\Scripts\python.exe -B -X utf8 tools/contract_experiment.py --plan PLAN.json --status
.\.venv\Scripts\python.exe -B -X utf8 -m financial_rag.experiments.a --plan PLAN.json
.\.venv\Scripts\python.exe -B -X utf8 -m financial_rag.experiments.b --plan PLAN.json
.\.venv\Scripts\python.exe -B -X utf8 -m financial_rag.experiments.c --plan PLAN.json
.\.venv\Scripts\python.exe -B -X utf8 -m financial_rag.experiments.d --plan PLAN.json
.\.venv\Scripts\python.exe -B -X utf8 -m financial_rag.experiments.pre_d --plan PLAN.json --snapshot-plan EXISTING-D-PLAN.json
```

独立入口只执行自己的组别，含该组失败补跑，不执行其他组或评分；组别必须在冻结 plan 中。串行入口复用同一执行器，已成功的独立执行记录不重生成。pre-D 必须显式指定包含完整首次快照的 D 运行（`--snapshot-plan`），或通过 `--snapshot-material` 指定保留的复用包，逐题核对源库身份、题文、索引身份与快照结构；缺失在任何 API 请求前停止，不隐式启动 D。本次 ABC 正式计划不允许 D/pre-D 入口启动。

## 共同评分与报告

```powershell
.\.venv\Scripts\python.exe -B -X utf8 tools/experiment_score.py --plan PLAN.json
.\.venv\Scripts\python.exe -B -X utf8 tools/experiment_report.py --plan PLAN.json
```

串行驱动自动调用共同评分、来源核查及汇总。评分只读取新运行库的冻结组别，复用库不参与 judge。报告重建不调用模型、不修改分数。新ABC先通过 `review/` 核查，再与复用材料生成 `delivery/REPORT.zh-CN.html`、逐题CSV及文件清单；只有这时切换 `.local/current-delivery.json`。

`finance-rag run/evaluate/report`仍是公共执行/评分/汇总实现的模块级入口；正式交付使用冻结计划入口，避免缺少冻结和来源核查。清洗、enrich、embed 等上游CLI继续保留。

## 失败与恢复

首次全部新答案执行结束后，仅无有效答案者统一补跑最多三轮。已有效答案即使判错也不重新生成；四次尝试耗尽按错，固定分母。每轮保留成功记录状态检查点，全部尝试及物理请求在同一事务运行库保存，不再复制整库分轮备份。

中断先核实真实 PID、父子、创建时间、锁与提交进度，保全现场后恢复同一计划。正常只读 SQLite 检查包含已提交 WAL；不能删 WAL 绕过检查。只有确认进程已退出才处理残锁。Windows任务 Ready 不等于 Python 已退出；一次性任务无 Trigger、无自动重启、无限执行时限。持续鉴权/余额/无成功进展的网关故障停止并报告；少量可恢复429不反复启停。

费用按实际请求边界统计，judge以任务键关联；缓存不冒充新请求，未知用量不当零。原暂停在途请求的未知用量保留说明，实际货币账单不可见不虚构价格。

## 上游能力

事实清洗遵循 AGENTS.md 和 `.agents/skills/clean-financial-documents/SKILL.md`。`tools/build_contract_index.py`提供建库入口。当前 cleaning-v2 profiles 仅适用于对应文档版本；新文档先完成解析和事实验收，再进行分块、摘要与建库。

## 离线测试

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m pytest -m 'not docling' -q --basetemp .local/audits/local-tests -p no:cacheprovider
```

虚构材料与本地协议替身不调用外部模型。真实 Docling 权重集成测试独立标记，首次运行可能下载权重。测试覆盖组别隔离、pre-D 快照门槛、成功记录保护、SQLite/WAL 恢复、补跑耗尽、评分续评和复用记录不再评分。

项目不执行文件、产物或源码哈希校验，不生成校验清单；保留结构、类型、来源关系、文件可读性和业务不变量检查。内容 ID、缓存键与去重键仍可使用哈希。报告重建只读原始记录，不修改答案、分数或标签。
