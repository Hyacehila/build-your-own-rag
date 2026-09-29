# 全量实验准备与异地交接

[首页](README.md) · [设计](DESIGN.md) · [操作](OPERATIONS.md) · [当前验收](VALIDATION.md)

## 1. 结论与适用范围

**2026-09-29 更新：**下一轮代码已调整四组的证据处理、C 的增量增强、预算和重试。请先读[修复报告](REPAIR_REPORT_20260929.md)；本文后续描述保留原交接时点。不能用新的代码和 config 续跑旧 run ID，也不能把局部增强的诊断索引用于正式四组实验。朋友原运行的源码差异、完整增强缓存和逐题轨迹仍需一并归档。

2026-09-25 初次盘点确认可以进入全量实验准备阶段；2026-09-26 补齐源码交付。已发布事实和 A/B 可复用；剩余工作包括正式题集接入验收、真实 C/D 构建与验收、全量费用控制，以及新环境小样与 judge 验证。代码齐全不等于这些真实实验步骤已经完成。

推荐冻结当前 v2 为本轮基线。保留已记录的解析缺陷，不把尚未发布的 MinerU 定点候选混入本轮事实。如果决定采用新事实，先独立发布，再重新完成受影响的下游验收。不要用正式题目、答案或标签指导清洗或增强。

本文件是交接时点的盘点和执行约定，不替代持续维护的 [VALIDATION.md](VALIDATION.md)。下列“待验证”步骤不能解读为已经跑通；本次盘点没有启动付费 API 或正式实验。

| 层次 | 盘点结论 | 正式实验前的工作 |
|---|---|---|
| 事实发布及真实 A/B | 已有可恢复交接物与来源审阅记录 | 新环境复核哈希、恢复和检索 |
| C/D 控制流 | 有虚构小样验证 | 完成真实全库增强、向量和回读验收 |
| 正式英文题集 | 本机旧工作区有准备数据，发布库未绑定 | 按锁定版本下载或可靠迁移，再绑定和重建索引映射 |
| run/evaluate/report | 入口和离线测试已有 | 验证真实小样、judge、恢复续跑和完整性门槛 |
| Git 交付 | 数据库先行发布，本次补齐源码、锁文件、配置和文档 | 接收包含完整项目的代码提交，核对 LFS 对象 |
| 全量预算控制 | 有逐题预算与小样请求账本 | 配置可执行的全量请求/费用限制和中止机制 |

两个 SQLite、LFS 规则和验收记录先在提交 `03de8154c823327c3cbb19cd8c8fa2164ad107df` 推送到远端；该提交不含实验源码。接收方必须更新到包含本文件、`src/`、`tests/`、`pyproject.toml`、`uv.lock` 和配置的后续代码提交，不能停在 `03de815`。本次交付还包含项目 Skill、profiles 和 tools。代码结构及当前目录导航见[开发说明](DEVELOPMENT.md)，全量实验的前置验收仍需完成。

## 2. 盘点发现的具体缺口

1. **初次交付只有数据库，源码在本次补齐。** 初次盘点时本仓库 HEAD 为 `038c2b0`，实验目录尚未被跟踪；随后 `03de815` 只发布了数据库。接收方若缺少源码和锁文件，先核对 commit 并更新，不能判断为 LFS 拉取失败。LFS 规则存在也不等于对应对象已经下载到本机。
2. **运行版本必须是 Python 3.11。** `.python-version` 是 3.11，项目要求 `>=3.11,<3.12`。本次已修正操作手册里的 3.12。文档 CI 使用 3.12 只运行独立的文档检查脚本，不能据此认定实验环境支持 3.12；当前 CI 也没有跑实验测试。
3. **题集接入会影响索引身份。** `dataset.py` 的发布库回退为 `document_only`；`run` 明确拒绝它。`download` 能构建正式题集，但它会写入新的 dataset ID，`current_index` 随即将旧 A/B 判为 stale。本次用只读工作库、内存替换 metadata 的方式复现了两组错误，没有修改原库。不要手改索引 ID 来绕过检查；应在独立工作区接入题集，重新 chunk，并根据相同文本及模型身份复用向量。现有单元测试覆盖了无题集的发布恢复，未覆盖真实发布接正式题集再完成全流程。
4. **真实 C/D 仍待构建。** `enrich` 支持逐资产缓存，`chunk --index enriched` 要求所有活动图片/表格均有增强，D 复用 C 的索引。必须验证增强来源、全部切片向量覆盖和真实原页回读；目前 `doctor` 的索引检查仅遍历 flat/structured，`retrieval_release check` 也是 A/B 验收，不能单独证明 C/D 完整。
5. **小样限额不适用全量。** `Validation.deepseek_request_limit` 最大允许 60；未配置 ledger 时没有全局请求上限，且账本不计 embedding。`estimate-embedding` 不估算 chat、vision、judge 费用。正式实验需有覆盖各角色、重试及续跑的预算方案，不能简单取消小样限制后无人值守。
6. **API 和 judge 要在朋友环境重验。** 既有 live preflight 不调用 judge，且固定读取 `config.toml`。`doctor --live` 需要 ledger，并有成功结果缓存；独立小样工作区可避免把旧记录当成本机的新验证。模型名称、接口和参数是本项目配置契约，不能以“现在能用的另一模型”静默替换后继续复用旧向量。
7. **命令返回成功不等于完整评测。** `evaluate` 可在 judge 出错时返回带 `pending` 的结果，`report` 仍可生成报告。必须验收记录数、失败数、pending、四组题目集合及 `full_benchmark`，不能只看退出码或报告文件存在。
8. **当前 export 不是完整实验备份。** `retrieval_release export` 按 A/B 发布范围裁剪索引，并清除 cache、runs、records、api_calls。完成 C/D 或实验后，不得把它当全量结果归档器。需要单独保存完整工作数据库、增强缓存、题集文件、运行目录和用量记录。

## 3. 你需要交给朋友什么

| 交付项 | 具体内容 | 交付方式 |
|---|---|---|
| 代码版本 | 仓库地址、已推送 branch/tag 和完整 commit SHA | Git；必须含本目录隐藏文件、Skill、profiles、tools、tests、uv.lock |
| 两份发布库 | `handoff/finance-facts-v2.sqlite` 和 `handoff/finance-retrieval-v2.sqlite`，以及两个验收 JSON | Git LFS；也可另行传文件并按下表验 hash |
| 原 PDF | 六个原文件，不能用网页新下载的同名新版替代 | 单独压缩传输；或从配置指定的 HF revision 下载并校验 |
| 数据集来源 | `config.toml` 中固定的 repo/revision、英文过滤和物理页映射规则 | 代码；正式 questions/labels/mapping/manifest 应在目的环境重建或经验证迁移 |
| 模型凭据 | 朋友自己的 `DASHSCOPE_API_KEY`、`DEEPSEEK_API_KEY`，模型访问权限、额度及限流信息 | 环境变量或本地 `.env`，不要进 Git、聊天日志或结果包 |
| 实验约定 | v2 基线、A/B/C/D、模型身份、检索/阅读预算、judge 规则、费用限额、失败策略 | 填写下方提示词；运行前冻结配置 |
| 结果回传约定 | 完整运行产物、失败清单、用量和恢复说明 | 独立结果归档；不要只发汇总表或截图 |

只做本轮实验不要求 MinerU token、Docling Studio 或重跑神经解析。模型服务在远端，复用发布物的路径不以本地 GPU 为前提；RAM、耗时及磁盘峰值应在目的机器实测。两份 SQLite 合计约 1.84 GB，六份原 PDF 合计约 27.49 MB；还要容纳 LFS 本地对象、恢复副本、Python 依赖、增强和结果，不能按下载体积估算全部磁盘需求。不要传 `.venv` 或整个历史 `.local`。

本机可交付的原 PDF 位于项目 `.local/retrieval-v2/raw/`。旧工作区 `.local/finance/datasets/` 存在准备好的英文题集；它与发布库的文档版本集合在本次检查中一致，但其路径需要迁移，且尚未完成与当前发布的正式运行绑定。直接传该目录不等于已解决接入问题。

### 发布文件校验值（2026-09-25 实测）

| 文件 | 字节数 | SHA-256 |
|---|---:|---|
| finance-facts-v2.sqlite | 767737856 | `4ddc69323a6f93cb56758962a0575388ee054a0b4d396983134589405b1aaa03` |
| finance-retrieval-v2.sqlite | 1075511296 | `b95dd62971ff4eef648a7a862bca5fd23715ec8d8e780690cf645753fd385d29` |

| PDF | 物理页数 | SHA-256 |
|---|---:|---|
| bank_of_america_2024.pdf | 305 | `383c9e0653bf670f167e85ef1ee31b0d97d1187f1cb8d70fff1a051aee59cac8` |
| citigroup_2024.pdf | 963 | `4abb08e3fc603f4ce6aa090bade8ec2e0f31885507ff178d0ee84321c5f85213` |
| goldman_sachs_2024.pdf | 614 | `6a4bd9e19660a70fb002fe8ac6f34ed2ae56dc4c09d94af83f0e5b1716ef2c51` |
| jpmorgan_chase_2024.pdf | 437 | `7539ad1d5e56a65a5681d3ef35bf08bcc0b8d9545629c42d04500c5bf4329caf` |
| morgan_stanley_2024.pdf | 268 | `8dce0678cf23a42b0b761a079a2cd345e35d6132a54eeaf693d8ec128f200187` |
| wells_fargo_2024.pdf | 355 | `6b34a54af6f172aa73bcc3338d2108bbcdd858ddc4eb15b80ed5bd07782b70ab` |

### 发布者与接收者检查

发布者先检查待提交清单，仅提交本次要交付的文件；不要把 `.env`、原 PDF、工作数据库或不相关的根目录改动一并加入。使用现有 LFS 规则跟踪两个大文件，推送后以一次干净克隆验证代码、LFS 对象、文档与离线测试。提供最终完整 commit SHA；现在的旧 HEAD 不是可用交付版本。

接收者在仓库根目录执行（先把占位符替换为真实值）：

```text
git clone https://github.com/Hyacehila/build-your-own-rag.git
cd build-your-own-rag
git lfs install
git checkout <交付的完整commit_SHA>
git lfs pull
git lfs fsck
git rev-parse HEAD
git status --short
cd examples/financial-document-rag
uv python install 3.11
uv sync --locked
uv run --locked python --version
uv run --locked pytest -q -m "not docling"
uv run --locked python tools/check_docs.py
```

SQLite 必须是上表大小和哈希的二进制文件，不能只是 LFS pointer。若朋友已有仓库，fetch 后切到交付 commit，保留其已有改动。不要 `git clean` 或硬重置来清空环境。Windows 可用 `Get-FileHash -Algorithm SHA256`；其他系统使用等效 SHA-256 工具。

## 4. 目的环境的分阶段运行

以下是执行顺序与验收条件，不是一条未经检查的全量脚本。命令以项目目录为当前目录；`finance-rag` 的全局 `--config` 必须放在子命令前。每步检查退出码及 JSON 结果后再继续。

### 阶段一：恢复已发布基线，不调用模型

复制 `config.toml` 为被忽略的 `config.local.toml`，仅把 `work_dir` 改为全新隔离目录，例如 `.local/full-experiment-v2`。先 prepare，再 doctor；不要先对空目录运行会创建数据库的命令。

```text
uv run --locked python -m financial_rag.retrieval_release prepare --config config.local.toml --database handoff/finance-retrieval-v2.sqlite --pdf-dir <原PDF目录>
uv run --locked python -m financial_rag.retrieval_release check --config config.local.toml
uv run --locked finance-rag --config config.local.toml doctor
```

确认事实 release ID、两组切片和向量符合发布记录，来源回读正常，没有神经解析或模型 HTTP 调用。需要独立检查事实库时，使用操作手册中的 cleaning restore/check，输出到另一个新目录。

### 阶段二：固定正式题集，完成接入验收

优先按锁定 revision 执行 `download`，自动准备英文问题、参考答案及页级标签。此步骤会访问 Hugging Face、下载数据，不是付费推理；可能下载 corpus parquet 等超出 PDF 的数据。网络受限时，用经过相同哈希及映射验证的本地数据包迁移，不能手填题目或相关页。

```text
uv run --locked finance-rag --config config.local.toml download
uv run --locked finance-rag --config config.local.toml chunk --index flat
uv run --locked finance-rag --config config.local.toml chunk --index structured
uv run --locked python -m financial_rag.retrieval_release reuse --config config.local.toml --database handoff/finance-retrieval-v2.sqlite
uv run --locked finance-rag --config config.local.toml estimate-embedding
```

这是由现有入口组成的接入路线，真实组合尚需验收。`reuse` 会补向量映射，应核对 A/B 全覆盖、文本和模型身份完全一致；不能只看 copied 数，已在工作库的向量可能无需再次复制。若仍有未命中，先查清原因，再在获准的 API 预算内 embed，不能默认接受大批重算。重新运行 A/B check 和 doctor；记录新 dataset/index ID 及其与 v2 事实的对应关系。

接入的硬性检查：正式英文问题数量及唯一性满足配置；questions 和 labels 一一对应；每题有正相关页；doc_version、PDF 哈希、物理页映射、题集和标签校验值均有效；runtime 的 queries 不含参考答案或 qrels。禁止为了继续运行而把 document_only 改成 false 或删掉 stale 检查。

### 阶段三：API 小样、真实增强与 C/D 验收

用独立配置、独立 ledger 进行有限协议小样，验证 embedding、正文回答、图像输入、工具调用和 judge；不沿用旧 preflight 的缓存结论。真实资产小样使用 `enrich --node-id <活动节点ID>`，按文档和资产类型选择，不能根据正式标签挑选“容易出分”的样本。

在小样通过且预算控制落实后执行：

```text
uv run --locked finance-rag --config config.local.toml enrich
uv run --locked finance-rag --config config.local.toml chunk --index enriched
uv run --locked finance-rag --config config.local.toml estimate-embedding
uv run --locked finance-rag --config config.local.toml embed --index enriched
```

验收全部活动资产的增强 key、失败清单、切片及向量映射、模型身份、来源与原页回读，明确 D 使用同一个 enriched index。从六份文档抽取正文、表格、图片场景检查，不能只用虚构 fixture。内容不清楚时必须保留不确定性；增强文本不能充当原始证据。

规模参考：本次只读盘点的当前清洗版本有 125 个 picture 和 1,575 个 table，即 1,700 个增强对象。一次成功新建每对象至少一次生成调用；长表分段、合并与重试会增加请求。该数量不能从数据库全部历史 nodes 汇总，必须只统计活动清洗版本。

全量四组预期 309 × 4 = 1,236 条回答记录。当前 A/B/C 每题一次回答调用，D 每题最多 6 次；单次首次全量在逐题预算下至多 2,781 次回答模型调用，未含增强、嵌入、judge、额外运行或错误重试。judge 未命中缓存时最多为每个成功答案增加一次逻辑评分，HTTP 重试另计。不要把这些调用数换算成没有依据的费用或预计完成时间。

### 阶段四：真实四组小样，再正式全量

```text
uv run --locked finance-rag --config config.local.toml run --run v2-pilot-001 --arms A B C D --limit 3
uv run --locked finance-rag --config config.local.toml evaluate --run v2-pilot-001
uv run --locked finance-rag --config config.local.toml report --run v2-pilot-001
```

`--limit` 只选题集前 N 题，不保证覆盖六份文档及资产类型，需结合上一阶段的定向来源检查。pilot 是链路门槛，不足以推断效果。先确认真实 C/D、judge、引用、失败检测与账本工作正常，再启动正式 run。

```text
uv run --locked finance-rag --config config.local.toml run --run v2-full-001 --arms A B C D
uv run --locked finance-rag --config config.local.toml evaluate --run v2-full-001
uv run --locked finance-rag --config config.local.toml report --run v2-full-001
uv run --locked finance-rag --config config.local.toml usage
```

以上 ID 是示例。pilot 与 full 必须不同；同一个 ID 绑定代码、数据、配置、arms 和题目选择，不能先 limit 3 再去掉 limit 沿用同名 run。中断后用完全相同的 full 命令续跑，已成功记录会跳过；重试失败记录时增加 `--retry-errors`，保留旧尝试并遵守预先约定的重试上限。运行中不要修改代码、索引或模型配置，不要并发多个写进程操作同一个工作库。

SQLite 按题保存，`results.jsonl` 每组结束时才导出；中途 JSONL 不齐不等于丢题。要备份活跃库，使用 SQLite backup；若直接复制文件，先停写、关闭连接并妥善处理 WAL。保留 snapshots/raw/datasets/runs 等恢复需要的文件与路径说明。

## 5. 正式完成的验收与回传

- `manifest.json` 标明 `synthetic=false`、`full_benchmark=true`；四组各有相同的 309 个唯一 query ID，总记录为 1,236。
- 理想完成条件是全部运行记录 `status=ok`、judge `pending=0`，且所有应评分答案都有有效评分。若有无法排除的失败或缺标签，应明确宣布“部分完成”，报告分母、错误及缺失原因，不能用有分数的子集冒充全量准确率。
- 检查 citation 格式/来源绑定及错误分布。引用有效性只说明来源 ID 可回溯，不证明答案事实正确；另抽查原文与 judge 结论。
- 回传 run 目录中的 `manifest.json`、`results.jsonl`、`scored.jsonl`、`evaluation.json`、`summary.csv`、`deltas.csv`、`report.md`；另附 `api-usage.json`、运行日志、失败清单、预算账本和阶段验收 JSON。
- 归档完整工作库与增强缓存、正式数据清单及文件哈希、脱敏配置、代码 commit、Python/uv/系统信息、实际模型身份、开始结束时间、续跑/重试记录，并在报告中关联事实 release ID、三种 index ID。凭据不得进入归档。
- 报告同时看检索、回读覆盖、引用和答案质量，以及模型调用数、用量和耗时。A→B 包含解析及切片方案变化；C→D 的初始检索共用索引和搜索，应重点解释阅读与回答变化，不能把初始排名说成独立的 Agent 检索提升。
- 当前 judge 与回答配置为同一模型，首轮可按冻结协议保留并披露这一限制；发布强效果结论前，建议独立人工复核抽样与配对不确定性分析。不要一边看正式结果一边改变 judge 规则。

## 6. 朋友给 AI 的启动提示词

先填写方括号里的内容；未填的费用授权不能被 AI 自行猜测。该提示词允许 AI 完成接收检查及必要的运行准备，不意味着当前所有门槛已经通过。

```text
请接手仓库 examples/financial-document-rag 的异地全量实验，目标是基于已发布 v2 事实，完成英文 309 题 A/B/C/D 四组运行、judge 评分、报告和可恢复结果归档。

交付 commit：[完整 SHA]
原 PDF 目录：[本机绝对路径]
实验工作目录：[全新绝对路径或项目内 .local 子目录]
凭据：[已在本机环境变量或 .env 配置，仅检查是否存在，禁止显示值]
API 费用授权：[允许的总额、币种、平台硬额度/可执行限制]
请求或 token 限额：[按阶段/角色填写，含 embedding、vision、chat、judge 和重试]
失败重试约定：[最大轮数、连续故障中止条件]
结果目录：[路径]

先阅读 AGENTS.md、README.md、DESIGN.md、VALIDATION.md、OPERATIONS.md 和 EXPERIMENT_HANDOFF.md，按当前代码核对说明。执行以下阶段，持续保存进度及证据：

1. 核对 commit、Git LFS 实体和发布库/PDF 哈希，使用 Python 3.11 与 uv sync --locked，运行离线测试。保留原发布文件，恢复到全新工作区。验证 A/B 和 v2 来源链，不重新做全库神经解析。
2. 按 config.toml 的锁定 HF revision 准备英文题目和页级标注，核对所有文档版本、唯一题目、页码与文件哈希。发布库原本是 document-only，接入题集会改变 dataset ID。使用已有入口重新 chunk 并复用向量，完成真实组合验收；不能修改 metadata 来跳过 stale 或来源检查。清洗和增强不得使用答案或标签。
3. 落实上述已授权的全量预算限制，再做新环境 API 协议、真实资产和 judge 小样。现有 validation ledger 最多 60 次且不计 embedding，只用于小样，不能冒充全量保护。预算缺失时完成离线工作并列出需要我补充的具体信息，不发送付费请求。
4. 在预算内构建真实全库 C 的增强、切片和向量，D 复用 C。检查全部活动资产、完整向量覆盖和原文/原页回读。A/B check 或虚构小样通过不能当成真实 C/D 通过。
5. 用独立 pilot ID 跑真实四组小样及 judge。验证成功后，在已授权范围内直接用新的 full ID 跑正式四组，不必重复询问已经授权的阶段。不要将 --limit 运行与完整运行复用同一个 ID。中断使用同一配置/代码/选择续跑，按约定有限重试并保留失败记录。
6. 正式验收四组各 309 个相同唯一 query ID、总计 1,236 条记录、synthetic=false、full_benchmark=true、运行失败数和 judge pending。evaluate/report 有文件或退出码为零不代表全量成功。部分失败必须清楚报告，不能丢弃失败题来抬高准确率。
7. 回传交接文档要求的所有结果、费用、配置、版本与来源证据。使用完整工作区归档/SQLite backup；retrieval_release export 只适合 A/B 发布，不能拿它备份 C/D 和实验结果。

常规环境与路径修复可自行完成并记录。需要新增题集适配、全量预算控制或 C/D 完整性检查时，先完成最小实现及针对性离线测试，再运行依赖它的付费步骤；冻结最终代码和配置后创建正式 run。不要静默更换模型、解析器、事实发布或评测标准；遇到不兼容时给出明确阻塞与选项。不要改写 .env、删除已有结果、硬重置仓库或启动无限重试。最终给出能复现的命令和阶段状态，不只返回计划。
```

## 7. 下一开发阶段的完成标准

建议先在开发环境收尾，再把朋友定位为实验执行者，减少边花费边修管线：

1. 固化“发布恢复 → 正式题集接入 → A/B 向量复用”的集成验收，最好提供单一接入命令或受检查的启动脚本；不需要重做事实层。
2. 增加覆盖真实 C/D 的完整性检查，以及各阶段请求/用量限制、错误中止和恢复验证；不能只扩大原来的 60 次上限。
3. 将非神经解析的实验测试纳入 CI，完成一次干净环境接收演练；校验 LFS、输入清单与脱敏后的结果归档。
4. 固定本轮费用与评测协议，提交交付代码，发出 commit 和输入包，再由朋友完成真实增强、小样与全量。

本次盘点运行了 `uv run --locked pytest -q -m "not docling"`：68 passed，1 deselected，66 条弃用警告；未运行真实神经解析器集成测试。`ruff check src tests tools` 和文档检查通过；当前真实工作库的 `retrieval_release check` 再次通过，混合检索使用已有缓存向量，没有新模型请求。两份发布库 SHA-256 已实测；正式题集接入后的真实全流程、新环境模型兼容性和正式答案质量仍须按上述门槛验证。
