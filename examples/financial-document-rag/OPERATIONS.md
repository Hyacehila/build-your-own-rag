# 运行、评分与交付

所有命令在项目根目录执行，使用 `.venv/Scripts/python.exe -B -X utf8`。环境采用 Python 3.11、`uv.lock`；本机保留 OCR 可选依赖。环境检查：`uv sync --locked --check --offline --extra ocr`；安装使用 `uv sync --locked --extra ocr --link-mode copy`。

## 配置与准备

### 网关与凭据

`config.toml` 的 chat、vision、judge 使用 `base_url_env = "FINANCIAL_RAG_BASE_URL"`。读取配置时从进程环境或配置文件同目录的 `.env` 解析，进程环境优先；缺失或不合法时立即停止，不发送请求。地址必须是无凭据、无查询参数的 HTTP(S) API URL，包含实际 API 前缀。embedding 地址仍使用原提供商设置。

`.env.example` 仅列变量名和说明。已有 `.env` 不要覆盖；密钥填写 `NEW_API_KEY`、`DASHSCOPE_API_KEY`，可选 MinerU 功能另用 `MINERU_API_KEY`。不要提交真实 `.env`。

新计划的 `experiment.toml` 保存已解析的 `base_url`，不再保存 `base_url_env`。已有计划中的固定地址不受环境变量影响；两种字段不能同时配置。相同实际地址、模型和参数继续得到相同模型身份；改变真实地址属于新运行配置，不用于覆盖旧计划。

五组名称统一为 A、B、C、pre-D、D。生产配置 `config.toml`：RRF20、两路各50；A 最终选择10个入口，B 最多选择20个原始入口，C 保留 B 的原始入口后增加最多5个摘要入口。B/C六阶段25k/8图，受控原页最后最多4k，不预留5k跨小节额度。`config.offline.toml`不配置外部模型服务，用于本地检查。

容量来自 `config/gateway-capacity.json` 和原响应证明：131072上下文，16384输出预留，文本/图像代理计数乘1.2；不是准确计费。原文及索引只读复用，不因代码整理重建。

容量文件中的地址是历史证明的绑定对象，不是环境变量的备用连接地址。使用另一网关时，须另行准备与实际地址、模型匹配的容量文件，并设置 `query.capacity_evidence_path`；不要修改原证明来冒充新网关的实测能力。当前检查会拒绝绑定不符的证明，不会自动发起容量探针。

## 输入准备与适用范围

| 使用场景 | 所需输入 | 可用入口 |
|---|---|---|
| 查看代码、离线验证 | Python 3.11、锁定依赖 | `financial_rag.cli --help`、pytest；无需真实API和`.local`历史材料 |
| 处理自己的文档 | 原始PDF、文档清单、对应清洗决策与profiles | 清洗技能、`financial_rag.cleaning`、`financial_rag.final_outline` |
| 新建检索实验 | 已验收事实库、三套索引与向量、原题及标签、容量证明 | `chunk`/`enrich`/`embed`、`experiment_setup.py` |
| 复核当前历史结果 | 原source、cohort、运行库，以及D/pre-D独立复用包 | 既有plan和`experiment_report.py`；不重新评分 |

自己的文档处理顺序是：准备文档清单与原始解析 → 按清洗技能构建候选、人工审阅标题与来源 → 验收并发布事实库 → flat/structured 分块与向量化 → 摘要与 enriched 索引 → 准备题集 → 创建独立实验。六份财报的 profiles 对应各自的文档版本，不能直接套用到新文档。

当前 CLI 面向清单和工作区，尚无任意 PDF 路径的一键导入；`download` 针对配置中的 [ViDoRe V3 Finance - EN](https://huggingface.co/datasets/vidore/vidore_v3_finance_en/tree/main) 数据集。各入口参数可通过以下命令查看，这些帮助命令不创建实验或调用 API：

```powershell
.\.venv\Scripts\python.exe -B -m financial_rag.cli --help
.\.venv\Scripts\python.exe -B -m financial_rag.cleaning --help
.\.venv\Scripts\python.exe -B -m financial_rag.final_outline --help
.\.venv\Scripts\python.exe -B tools/build_contract_index.py --help
```

正式准备器使用已准备的原文和清洗结果。默认查找以下本地材料，缺少时会提示路径与下一步：

- `query.source_work_dir/experiment.sqlite`：可读事实库及完整的flat/structured/enriched索引、对应embedding；PDF和原图路径也须有效。
- `--cohort`：`queries.jsonl`、`labels.jsonl`、`manifest.json`，包含有序题号和源数据集身份；原标签不变。固定历史296题的派生清单不随Git发布，也不由准备器自动猜测生成。
- 预实验额外需要 `--selection`：显式固定题号清单；全量使用整个cohort。
- 仅合并历史D/pre-D时需要 `--reuse-material`：原记录、原评分、快照与来源清单；没有复用包可以省略该参数，只运行新组别，不能填入历史分数代替缺失记录。

题量、文档量和页数由配置及清单核对。作者从原始题集中筛除部分有问题的题目，固定保留 296 题；现有下载准备流程不会自动重建这份筛选清单，直接使用固定版本的上游数据会因题数不符而失败。更换文档或问题范围须显式准备匹配配置，不混入当前历史运行。

本轮固定题集、清洗决策及历史投入无法仅凭代码仓库重建。复核材料由持有者另行交付，当前仓库不提供自动下载历史库的方式。

## 创建新计划

以下命令要求前述材料已就绪。`--check`只进行本地读取检查，不等于验证API连通性。

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

本地 `doctor` 检查 SQLite、外键、已准备的数据与索引、模型配置和预算状态，不调用模型服务，并在工作目录写入 `doctor.json`。报告中的 `dataset_check` 为 `not_prepared` 时，表示没有附加题集，题目、来源及相关索引检查未执行；`passed` 和 `failed` 分别表示已完成检查或发现问题。该检查不证明外部服务可用。

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m financial_rag.cli --config config.offline.toml doctor
.\.venv\Scripts\python.exe -B -X utf8 -m pytest -m 'not docling' -q --basetemp .local/audits/local-tests -p no:cacheprovider
```

测试使用临时输入和进程内协议替身，不调用外部模型。真实财报与 Docling 权重的集成测试独立标记：设置 `FINANCE_TEST_PDF` 指向上游摩根大通年报，并设置 `FINANCE_DOCLING_ARTIFACTS` 指向本地权重目录，然后把测试命令中的 `-m 'not docling'` 换成 `-m docling`；缺少任一材料时自动跳过。测试覆盖组别隔离、pre-D 快照门槛、成功记录保护、SQLite/WAL 恢复、补跑耗尽、评分续评和复用记录不再评分。

验收依据结构、类型、来源关系、文件可读性和业务不变量。既有内容 ID、缓存键与去重键保持兼容。报告重建只读原始记录，不修改答案、分数或标签。
