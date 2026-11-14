# 金融文档 RAG

[返回示例目录](../README.md) · [主线教程](../../README.md#主线教程目录)

这是主线前六章的配套项目，以六份真实金融财报及其问答题为材料，展示从长篇 PDF 到可追溯原文的 RAG 问答流程。项目提取 PDF 文本，并使用 Docling 解析标题、正文、表格和图片，保留内容与原始页面的关联；解析结果还可以进一步清洗和审阅，对局部难页使用 MinerU 辅助诊断。

项目分别按平面文本和文档结构切片，也可以为正文与图表生成摘要，再通过关键词和向量混合检索定位相关内容。查询时，系统按需回读上下文、表格、图注和原始页面，将回答关联到读取过的来源；读者可以对比 Native RAG 和 Agentic RAG 两种方式。项目还使用从上游数据中筛选出的 296 道问题进行评测，记录逐题回答、引用和评分，帮助读者观察不同文档表示、检索入口和阅读方式的效果。

## 项目如何工作

教程分别介绍了文档格式、解析、切片、检索和原文回查；这个示例将它们接成一条完整链路：

![金融文档 RAG 项目流程：财报经过平面或结构化解析，结构化事实可选清洗后用于 B、C、pre-D 和 D](assets/financial-rag-pipeline-journal-v2.png)

两种解析路线服务于不同的比较：平面路线提供一个容易理解的基线，结构化路线保留标题、节点关系和来源位置。图中 B 指 C 复用的原文检索入口，并非将 B 的回答输入 C；摘要命中后仍回到原始内容回答。各方案使用关键词与向量混合检索，并对回答和来源进行评测。

| 方案 | 检索与阅读方式 |
| --- | --- |
| A | 检索平面原文切片 |
| B | 检索结构化原文，结合文档关系扩展上下文 |
| C | 在 B 的原文入口上补充摘要召回，再回读原文 |
| pre-D | 使用 D 自动扩展前的首次检索与阅读快照直接回答 |
| D | 在首次快照基础上自动扩展原文，再由模型按需继续检索和回读 |

下面按数据流介绍各环节，以及实现中为了处理真实财报所做的取舍。

## 数据集来源

项目使用 [ViDoRe V3 Finance - EN](https://huggingface.co/datasets/vidore/vidore_v3_finance_en) 数据集，其中包含六份美国金融机构 2024 财年的年度报告、英文问题、参考答案和相关页面标注。原始 PDF 可以在上游的 [pdfs 目录](https://huggingface.co/datasets/vidore/vidore_v3_finance_en/tree/main/pdfs)查看和下载。

为保持数据一致，示例下载时使用上游的 [7f432c1 版本](https://huggingface.co/datasets/vidore/vidore_v3_finance_en/tree/7f432c176d82e27546501ad8064a713ac3071809)，对应参数记录在 [config.toml](config.toml) 中。数据集标注与元数据采用 CC BY 4.0，原始财报及解析文字遵循各自来源的使用条件。

为了减少题目本身的问题对评测的影响，我从原始英文题集中筛除了 13 道题，它们在上游数据中的 `query_id` 分别为：`0`、`1`、`54`、`56`、`80`、`81`、`82`、`90`、`133`、`166`、`180`、`192`、`257`。剩余的 296 道题保留原题、参考答案和相关性标注，作为本项目统一的评测集。

## 文档解析与清洗

### 先保留原文和结构

平面路线使用 PyMuPDF 提取每页文本，再按长度切片；它适合做基线，但章节关系、表格结构和图片内容很难仅靠文字还原。结构化路线使用 Docling 解析 PDF，保留标题、正文、表格、图片、图注、页码和区域位置，并将节点及其父子关系写入 SQLite。后续切片和回读都引用这些节点，遇到问题时可以回到原始 PDF 检查。

这批财报主要有可用的文字层，默认使用 CPU、关闭 OCR 并启用较精细的表格识别；扫描件或文字层受损的页面可能需要 OCR。解析质量还取决于版面：多栏阅读顺序、跨页正文、合并表头和图片区域都可能出错，因此这里保留原始解析快照，并提供样页诊断、清洗候选和标题树审阅，而不是把一次解析的结果直接当作无误的原文。

### 用 MinerU 处理局部难页

Docling 的解析并非在每种版面上都理想。遇到漏表、表格错列或正文顺序混乱时，可以先用 `diagnose parse-sample` 比较本地解析配置；仍有问题时，使用 MinerU 对具体页面生成另一份结构化候选，再对照原页判断是否采纳。候选会记录原始文档和物理页映射；标题层级仍需结合目录和原文审阅，不能只凭解析器输出确定。

`diagnose mineru-prepare`、`mineru-submit`、`mineru-fetch` 和 `mineru-import` 提供最多 8 页的问题页对照，导入结果保存在独立目录，不会自动替换正在使用的事实。清洗流程也提供按诊断结果准备、提交和转换候选的入口；构建清洗文档时，代码会尝试将符合条件的表格候选合入，并记录接受或拒绝的原因。两条入口的配置和输出不同，操作前请看[项目适配说明](.agents/skills/clean-financial-documents/references/project-adapter.md)。当前转换器针对已支持的 MinerU 返回结构，并不是任意版本结果的通用转换器。

MinerU 云端调用需要 `MINERU_API_KEY`，会上传所选 PDF 页面；运行前应确认数据是否适合外发，并查看服务的当前额度和费用。本文的默认运行流程只使用本地 Docling，具体的清洗与接入步骤见下文。

## 切片与检索入口

平面切片按 token 长度和重叠量切分页文本。结构化切片使用 Docling 的 `HybridChunker`，结合事实层的标题路径与节点关系，尽量在同一章节内组织内容；表格使用多行表头序列化，较长表格拆分时重复必要表头。切片同时保存指向原始节点和页面的关系；片段可以唯一定位到原文时进一步细化来源，不能精确定位时保留较粗的来源范围。

财报中有些表格过长，有些图表则不包含可直接检索的文字。`enrich` 可以为连续正文、表格和图片生成摘要：正文按同一章节的连续切片组成窗口；长表格分段概括后再合并；图片及未提取出单元格的表格区域使用视觉模型查看原图。生成内容只作为补充入口，精确数字和引用仍来自回读的原文或原始页面。摘要、向量和评分结果可以缓存，已完成的问答记录也可以在续跑时复用；更改模型或解析结果后，受影响的内容需要重新生成。

正文摘要使用文本模型的非思考模式，单次输出上限为 1024 token，以控制批量摘要的开销；实现会向服务发送 `thinking` 和 `reasoning_effort` 参数。如果所用网关不支持这些参数，需要调整 [summary_windows.py](src/financial_rag/indexing/summary_windows.py) 中的摘要请求配置。

## 混合检索、回读与回答

示例用 SQLite FTS5 做关键词检索，用 `sqlite-vec` 做向量检索，再以 RRF 合并两路排名。关键词检索适合财报中的公司名、年份和术语，向量检索有助于匹配不同表述；两路的候选数、融合参数和最终返回数都可以在 [config.toml](config.toml) 中调整。切片及其来源、向量、运行记录放在同一个本地工作区，便于查看完整链路。

B 先按排名读取命中的结构化原文及其关联图注、脚注，再读取同章节的邻近节点；重复覆盖的原文不会反复占用候选位置。C 保留 B 的原文入口，再补充指向其他原文的摘要命中。表格或图片可以关联原始页面图像，便于模型结合文字阅读；启用图片回读时，回答模型也需要支持图片输入。文本与图片都有可配置的上限，避免一次性把整份财报塞进上下文。

固定流程不会根据正文中的 “Table 9” 等文字自动跳转，也不会自动扩展相邻章节或追加整页文本；这些情况可以由 D 继续搜索和指定页面回读。B/C 命中表格或图片时仍可读取其来源页面的图像。

D 使用结构化原文索引：先将问题整理成检索查询和待回答项，保存首次检索与阅读快照；随后程序自动扩展原文，再由 Agent 视情况继续搜索、回读上下文或查看原页。pre-D 直接使用同一份自动扩展前的快照回答。因此 pre-D 与 D 的差异同时包含自动扩展和后续探索，不能只归因于 Agent 的自主探索。程序保留读取过的证据、工具调用和引用关系；上下文空间不足时，较早的内容会被压缩成可再次读取的来源提示。上下文和图片 token 数采用估算，实际限制仍以模型服务为准。

## 评测什么

评测同时看检索和回答：NDCG@10、Recall@5、Recall@10 使用上游相关页面标注衡量初始检索，原页覆盖率统计实际读取的证据覆盖了哪些标注页；引用有效性检查引用是否指向已读取的证据。回答正确率由评分模型对照参考答案判断，逐题结果和理由都会保存。引用有效并不自动代表回答正确，模型评分也可能出错，因此有必要抽查原文和逐题记录。

## 环境与配置

项目使用 `uv` 管理 Python 3.11 环境和依赖。以下命令在 `examples/financial-document-rag/` 目录下执行：

```powershell
uv sync --locked --extra ocr
uv run --locked --offline --no-sync finance-rag --help
```

首次安装需要下载依赖；Docling 解析需要模型权重和计算资源，默认使用 CPU。`--extra ocr` 安装可选 OCR 依赖；默认财报解析配置关闭 OCR。首次使用 `tiktoken` 时也需要取得其编码文件。

将 [.env.example](.env.example) 复制为本机 `.env`（已有文件可直接编辑），填写 `FINANCIAL_RAG_BASE_URL`、`NEW_API_KEY` 和 `DASHSCOPE_API_KEY`。网关地址包含 API 前缀，例如 `https://your-gateway.example/v1`。默认配置使用该网关进行回答、图像描述和评分，使用 DashScope 进行向量化；请在 [config.toml](config.toml) 中按实际服务提供的模型 ID、地址和参数调整。下载数据和本地解析不需要模型服务密钥。

运行 D 方案前，按实际模型设置 `query.context_window_tokens` 和回答模型的 `max_tokens`。程序使用配置的 tokenizer 估算文本和图片请求大小，并为输出预留空间；这个估算不等于服务商的实际计费或精确 token 数。向量化、摘要、回答和评分会调用模型服务，可能产生费用。

阅读和检索的主要参数位于 `query` 与 `retrieval` 配置段：

| 参数 | 默认值与范围 | 作用 |
| --- | --- | --- |
| `sparse_k`、`dense_k`、`rrf_k`、`final_k` | `50`、`50`、`20`、`20`，均至少为 1 | 两路候选数、RRF 参数和最终入口数 |
| `summary_supplement_k` | `5`，范围 0–20 | C 在原文之外最多补充多少个摘要入口 |
| `bc_text_tokens` | `25000`，至少为 1 | B/C 的阅读文本上限，同时受 `text_tokens` 约束 |
| `page_images` | `8`，至少为 0 | 固定查询流程可读取的页面图像数 |
| `exploration_calls` | `24`，范围 1–100 | D 的探索阶段模型调用数，包含首次问题整理；结束时还会留一次最终回答机会 |
| `agent_total_images`、`agent_request_images` | `16`、`8`，范围均为 0–100 | D 总共可读取的页面图像数，以及单次请求中最多携带的图像数 |

如需限制模型服务的调用次数，可以在配置中增加一套可选预算：

```toml
[budget]
ledger = ".local/model-budget.sqlite"
batch = "example-1"
request_limit = 1000
token_limit = 1000000
unknown_usage_limit = 10
```

预算默认关闭；启用时必须设置 `ledger` 和 `request_limit`。同一个账本和批次统计向量化、文本、视觉和评分的所有 HTTP 尝试，包括重试；相同批次不能中途更改限额。`token_limit` 可省略，它依据服务商返回的用量在请求结束后累计，因此单次或并发请求可能越过该值；没有用量的请求会计入未知用量。MinerU 使用独立的服务，不计入这套模型调用预算。`doctor` 可以查看当前预算状态。

## 运行示例

以下步骤先以未经清洗的解析结果运行一题，再展示可选清洗和完整方案比较。也可以在结构化解析后、首次切片前完成清洗与接入，然后从接入后的切片步骤继续。

| 阶段 | 输入与主要命令 | 输出 | 是否必需及模型服务 |
| --- | --- | --- | --- |
| 数据与权重 | `download`、`download-models` | 财报、题集、标注和 Docling 权重 | 必需；需要下载，不调用问答模型服务 |
| 解析与可选清洗 | `parse`；可选清洗、审阅和 `attach-facts` | 平面或结构化事实，及可选的审阅事实层 | 解析必需；清洗可选，本地默认流程不调用模型服务；MinerU 云端诊断另计 |
| 切片与索引 | `chunk`、`embed`；C 还需要 `enrich` | 检索切片、向量，以及 C 所需摘要 | 所选方案的索引必需；向量化和摘要调用模型服务 |
| 回答与评测 | `run`、`evaluate`、`report` | 回答、来源、逐题评分和汇总报告 | 回答调用模型服务；评分调用 judge 模型 |

### 1. 准备数据和模型

```powershell
uv run --locked --no-sync finance-rag download
uv run --locked --no-sync finance-rag download-models
```

`download` 下载固定版本的数据，并按上述清单生成 296 题及原始标注；`download-models` 将 Docling 权重下载到 `model_weights/docling/`。这两步需要访问上游，但不调用问答模型服务。

### 2. 解析、建索引并运行一题

下面使用 B 方案检索结构化原文，运行题集中的第一题：

```powershell
uv run --locked --no-sync finance-rag parse --parser structured
uv run --locked --no-sync finance-rag chunk --index structured
uv run --locked --no-sync finance-rag embed --index structured
uv run --locked --no-sync finance-rag run --run demo --arms B --limit 1
uv run --locked --no-sync finance-rag evaluate --run demo
uv run --locked --no-sync finance-rag report --run demo
```

结果写入 `.local/workspace/runs/demo/`：`results.jsonl` 包含回答及来源，`scored.jsonl` 包含逐题评分，`summary.csv` 和 `report.md` 汇总评测结果。评分需要配置 judge 模型。可以用 `--query-id <题目ID>` 指定题目，多次传入以选择多题；`--limit` 选择题集前 N 题。

保留相同参数并重复使用运行编号会沿用已完成记录；需要重试失败题目时添加 `--retry-errors`。每个方案的每道题在同一运行编号下最多尝试 4 次（含首次），达到上限后会跳过；如需再次尝试，先排查失败原因，再用新的运行编号重新运行所选题目。更换题目、方案或配置时也要使用新的运行编号。

### 3. 可选：清洗并接入审阅后的事实层

初始解析可直接用于演示。如果希望修正标题结构、目录页或局部表格，可以在结构化解析后、切片前进行本节操作；这里放在一题演示之后，是为了让读者先跑通流程。先从现有原始解析准备独立的清洗工作区，再基于适用于当前文档和解析版本的 profile 构建候选：

```powershell
uv run --locked --no-sync python -m financial_rag.document.cleaning prepare --source-work-dir .local/workspace --workspace .local/cleaning
uv run --locked --no-sync python -m financial_rag.document.cleaning build --workspace .local/cleaning --profiles profiles/cleaning-v2 --output .local/reviewed-facts
uv run --locked --no-sync python -m financial_rag.document.cleaning check --output .local/reviewed-facts --no-chunks
uv run --locked --no-sync python -m financial_rag.document.final_outline packets --root .local/reviewed-facts --reviews .local/outline-review
```

[profiles/cleaning-v2/](profiles/cleaning-v2/) 中的配置只适用于对应版本的六份财报；使用前要核对文档、解析结果及节点引用。清洗和实际标题树审阅的方法见 [clean-financial-documents](.agents/skills/clean-financial-documents/SKILL.md) 与[项目适配说明](.agents/skills/clean-financial-documents/references/project-adapter.md)。`packets` 只生成审阅材料，需要对照目录和原页完成审阅，按说明为每份文档填写真实的 `receipt.json`，不能由程序代替审阅。

完成审阅后，验收并将事实层接入当前题集工作区：

```powershell
uv run --locked --no-sync python -m financial_rag.document.final_outline accept --root .local/reviewed-facts --reviews .local/outline-review
uv run --locked --no-sync finance-rag attach-facts --source .local/reviewed-facts
uv run --locked --no-sync finance-rag chunk --index structured
uv run --locked --no-sync finance-rag embed --index structured
```

接入时会核对验收记录、文档标识、页数和来源关系，并保留原题、标注及已有运行记录。结构化事实改变后需要重建相关索引；如果之前构建过增强索引，还要重新运行 `enrich` 和 `chunk --index enriched`。使用新的运行编号评测清洗后的结果。一个工作区只接入一份事实发布；需要比较另一版时，请使用新的工作目录。

如需先对问题页进行 MinerU 小样对照，可将下面的文档 ID 和物理页号换成实际选择的页面：

```powershell
uv run --locked --no-sync finance-rag diagnose mineru-prepare --page bank_of_america_2024:10 --bundle .local/mineru-trial
uv run --locked --no-sync finance-rag diagnose mineru-submit --bundle .local/mineru-trial
uv run --locked --no-sync finance-rag diagnose mineru-fetch --bundle .local/mineru-trial
uv run --locked --no-sync finance-rag diagnose mineru-import --bundle .local/mineru-trial
```

提交前先检查页面包和云端配置；结果尚未完成时稍后再次执行 `mineru-fetch`。这个小样只用于对照，不会自动进入清洗候选。需要在清洗构建中使用 MinerU 候选时，按项目适配说明使用 `financial_rag.document.cleaning` 的 `cloud-prepare`、`cloud-submit` 和 `cloud-fetch`，检查生成的页面清单及转换结果后再运行 `build`。

### 4. 比较五组方案

在结构化索引准备完成后，再构建平面索引和增强索引：

```powershell
uv run --locked --no-sync finance-rag parse --parser flat
uv run --locked --no-sync finance-rag chunk --index flat
uv run --locked --no-sync finance-rag enrich
uv run --locked --no-sync finance-rag chunk --index enriched
uv run --locked --no-sync finance-rag embed --index all
uv run --locked --no-sync finance-rag run --run comparison --arms A B C D pre-D --limit 1
uv run --locked --no-sync finance-rag evaluate --run comparison
uv run --locked --no-sync finance-rag report --run comparison
```

`enrich` 会为正文及图表生成摘要，需要文本和视觉模型。这里先用一题检查流程；去掉 `--limit 1` 后会运行完整的 296 题。pre-D 使用同一次 D 运行的首次阅读快照，因此需要与 D 一起运行。

处理自己的文档还需要准备文档清单、原始解析和对应的清洗决策；当前 `download` 入口针对本数据集。

## 代码与教程

| 内容 | 实现位置 | 相关教程 |
| --- | --- | --- |
| 数据下载和题集准备 | [dataset.py](src/financial_rag/dataset.py) | 全部章节 |
| 解析、事实层和清洗 | [document/](src/financial_rag/document/) | [001](../../tutorials/main/001-document-format.md)、[002](../../tutorials/main/002-docling-document.md)、[003](../../tutorials/main/003-ocr-and-parsing.md) |
| 切片、摘要和索引准备 | [indexing/](src/financial_rag/indexing/) | [004](../../tutorials/main/004-chunks-and-enrichment.md) |
| 检索、回读和回答 | [query/](src/financial_rag/query/) | [005](../../tutorials/main/005-retrieval-and-agentic-rag.md) |
| SQLite 存储与结构回查 | [storage.py](src/financial_rag/storage.py)、[readback.py](src/financial_rag/query/readback.py) | [006](../../tutorials/main/006-databases-and-source-lookup.md) |
| 评分与报告 | [evaluation.py](src/financial_rag/evaluation.py) | 评测 |

配置和模型调用分别位于 [config.py](src/financial_rag/config.py) 与 [api.py](src/financial_rag/api.py)；[tests/](tests/) 保存离线测试，`diagnostics/` 提供本地诊断。

## 本地检查

```powershell
uv run --locked --offline --no-sync finance-rag --config config.offline.toml doctor
uv run --locked --offline --no-sync python -B -m pytest -m "not docling" -q -p no:cacheprovider
```

`doctor` 检查本地配置、SQLite 和已准备的数据及索引，不调用模型服务，并在对应工作目录生成 `doctor.json`。真实 PDF 与本地 Docling 权重的集成测试使用 `docling` 标记，需要另外提供测试材料。

其他解析诊断和可选的 MinerU 入口可通过 `finance-rag diagnose --help` 查看。

原始财报、模型权重、索引、运行结果和真实密钥保存在本地，不随 Git 提交。

## 实验方案与结果

五组方案的检索与阅读方式见[项目如何工作](#项目如何工作)；检索参数见[环境与配置](#环境与配置)。

![296 题评测：A 190/296、B 210/296、C 212/296、pre-D 195/296、D 232/296](assets/financial-rag-results.svg)

回答正确率由评测模型判断。题集曾参与方案调试，这些结果并非盲测；方案间的差异也不能归因于单一因素。图中保留的是已有实验成绩；当前 B/C 的原文扩展流程已经简化，重新运行需要生成新的结果，不能把图中成绩当作新实现的测量值。
