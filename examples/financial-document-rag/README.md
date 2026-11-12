# 金融文档 RAG

五个实验组统一命名为 A、B、C、pre-D、D。A 使用平面原文检索，B 使用结构化原文，C 在 B 的原始入口上补充摘要召回，pre-D 使用首次未扩展快照，D 继续扩展阅读。读取数量与预算作为配置参数说明，不作为组名后缀。生产配置为 `config.toml`；`config.offline.toml` 不配置外部模型服务，可用于本地检查。

当前全量结果已完成：A 190/296（64.19%）、B 210/296（70.95%）、C 212/296（71.62%）；pre-D 195/296、D 232/296 为原批次复用。888 条新答案与 592 条复用记录合计 1,480 条，未重新生成或评分 D/pre-D。

最终报告见 `.local/current-delivery.json`。

## 配合教程阅读

[主线前六章](../../README.md#主线教程目录)使用本例的真实财报讨论文档表示、解析、切片、检索、回读和存储。本目录提供完整实现及实验说明。

| 教程 | 对应实现 |
| --- | --- |
| 001 · 事实层 | [facts.py](src/financial_rag/facts.py) 中的文档关系与事实投影 |
| 002 · DoclingDocument | [parsing.py](src/financial_rag/parsing.py) 中的结构化解析与节点提取 |
| 003 · OCR 与解析 | [parsing.py](src/financial_rag/parsing.py) 的解析配置，以及 [cleaning_profiles.py](src/financial_rag/cleaning_profiles.py) 的清洗处理 |
| 004 · 切片与增强 | [chunking.py](src/financial_rag/chunking.py) 与 [enrichment.py](src/financial_rag/enrichment.py) |
| 005 · 检索与回答 | [retrieval.py](src/financial_rag/retrieval.py)、[sources.py](src/financial_rag/sources.py) 与 [agent_reading.py](src/financial_rag/agent_reading.py) |
| 006 · 存储与回查 | [storage.py](src/financial_rag/storage.py) 与 [readback.py](src/financial_rag/readback.py) |

这些链接指向当前代码；历史实验当时的完整代码环境和运行材料不随仓库发布，阅读当前实现不能替代对历史运行的复核。

## 数据集来源

真实金融实验使用 [ViDoRe V3 Finance - EN](https://huggingface.co/datasets/vidore/vidore_v3_finance_en/tree/main) 数据集。作者从原始题集中筛除部分有问题的题目，固定保留 296 题，本例的全量实验均使用这组题目。上游页面提供下载入口；数据集标注与元数据采用 CC BY 4.0，原始财报及解析文字遵循各自来源的使用条件。数据源与固定版本见 [config.toml](config.toml)。

筛除的 13 道题在上游数据中的 `query_id` 分别为：`0`、`1`、`54`、`56`、`80`、`81`、`82`、`90`、`133`、`166`、`180`、`192`、`257`。这些编号是题目的 ID，不是文件中的行号；其余 296 题保留原题和标注。

已知第 `1755` 题的参考答案写作 “Valuation Committee Group”，而财报物理第 184 页写作 “Valuation Control Group”。本例保留上游参考答案和历史评分，复核该题时可对照原文。

## 首次使用

本仓库提供代码、配置、测试和清洗 profiles；`.local` 中的真实原文库、题集、运行记录与历史报告不随 Git 提供。复核历史成绩需要另行取得固定题集、清洗决策、运行库和复用材料，单凭下载上游数据集无法获得完整的复核材料。

`download` 使用固定版本的上游数据时，会按上述清单筛选并生成 296 题及其原始标注，同时生成可供实验准备器使用的题集清单。更换数据源或版本时，需按新数据显式设置 `dataset.excluded_query_ids` 和预期题数；默认的 13 个排除 ID 只适用于本例固定的数据源与版本。

在本目录安装锁定环境并运行离线检查：

```powershell
uv sync --locked --extra ocr
uv run --locked --offline python -B -m financial_rag.cli --help
uv run --locked --offline python -B -m pytest tests/test_current_configuration.py tests/test_portable_configuration.py -q -p no:cacheprovider
```

首次安装需要依赖下载；上面的检查不下载 Docling 模型，也不调用提供商 API。完整离线套件见 [运行说明](OPERATIONS.md#离线测试)。

需要真实模型时，将 `.env.example` 复制为本机 `.env`（已有文件不要覆盖），填写 `FINANCIAL_RAG_BASE_URL`、`NEW_API_KEY` 和 `DASHSCOPE_API_KEY`。网关地址包含 API 前缀，例如 `https://your-gateway.example/v1`。模型名称与实验参数仍由 `config.toml` 管理。

使用自己的文档需要先完成事实清洗和建库；本地文档处理与历史实验复核的输入要求见 [运行说明](OPERATIONS.md#输入准备与适用范围)。

|入口/目录|职责|
|---|---|
|`src/financial_rag/experiments/{a,b,c,pre_d,d}.py`|五个可独立运行的方案编排，每个支持 `--plan`|
|`runner.py` 与公共原文/API/存储模块|共享执行、请求审计、失败恢复；不复制五套基础实现|
|`tools/experiment_setup.py`|准备新计划、独立运行库和源码副本|
|`tools/contract_experiment.py`|按计划串行执行、统一补跑、评分与审计|
|`tools/experiment_score.py`|共同评分，仅评分本计划新生成的组别|
|`tools/experiment_report.py`|无模型调用的来源审计与报告重建|
|`tools/experiment_reuse.py`|原始 D/pre-D 材料提取及结构核查|
|`.local/reused/d-full296-20261008`|592 条复用记录、原评分、快照、关联请求及原版本凭据|
|`.local/contract-v1/source`、`cohort-296`|只读原文/图像/索引与固定题集原标签|
|`.local/experiments`|本次独立执行材料、检查点、结果与报告|

运行状态见 `.local/contract-v1/active.json`；经核查的最终交付指针为 `.local/current-delivery.json`。准备、运行和未完成报告都不会替代最终指针。

方案与边界见 [架构契约](DESIGN.md)，环境、执行、评分、报告及离线测试见 [运行说明](OPERATIONS.md)。

上游 PDF 清洗、Docling 解析、事实投影、分块、摘要和建库使用 `.agents/` 中的清洗技能及当前 profiles。原文、运行材料和密钥不在 Git 中；密钥只由环境或忽略的 `.env` 加载，不打包、不公开上传。
