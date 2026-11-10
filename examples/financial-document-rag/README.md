# 金融文档 RAG

五个实验组统一命名为 A、B、C、pre-D、D。A 使用平面原文检索，B 使用结构化原文，C 在 B 的原始入口上补充摘要召回，pre-D 使用首次未扩展快照，D 继续扩展阅读。读取数量与预算作为配置参数说明，不作为组名后缀。唯一生产配置为 `config.toml`；`config.offline.toml`仅用于虚构材料测试。

当前全量结果已完成：A 190/296（64.19%）、B 210/296（70.95%）、C 212/296（71.62%）；pre-D 195/296、D 232/296 为原批次复用。888 条新答案与 592 条复用记录合计 1,480 条，未重新生成或评分 D/pre-D。

最终报告见 `.local/current-delivery.json`。

## 配合教程阅读

[金融教学路线](TEACHING.md)从三页原创虚构报告出发，提供可直接阅读的 PDF、DoclingDocument 和文档树，也可在本地生成切片及原文回读结果。它不需要历史实验材料或 API，是理解前几章的起点；各章对应的实现位置也列在该页。

## 数据集来源

真实金融实验使用 [ViDoRe V3 Finance - EN](https://huggingface.co/datasets/vidore/vidore_v3_finance_en/tree/main) 数据集。上游页面提供下载入口并标注 CC BY 4.0，原始财报及相关内容还需遵循各自的使用条件；本例使用的数据源与固定版本见 [config.toml](config.toml)。历史实验的固定题集、清洗决策、运行库和复用材料另行管理，下载上游数据集不等于取得完整的成绩复核材料。

## 首次使用

本仓库交付代码、配置、测试、清洗 profiles 和原创教学参考材料。`.local` 中的真实原文库、题集、运行记录与历史报告不随 Git 提供；上面的成绩是已完成实验的记录，克隆仓库不会自动获得其复核材料。

在本目录安装锁定环境并运行离线检查：

```powershell
uv sync --locked --extra ocr
uv run --locked --offline python -B -m financial_rag.cli --help
uv run --locked --offline python -B -m pytest tests/test_current_configuration.py tests/test_portable_configuration.py -q -p no:cacheprovider
```

首次安装需要依赖下载；上面的检查不下载 Docling 模型，也不调用提供商 API。完整离线套件见 [运行说明](OPERATIONS.md#离线测试)。

需要真实模型时，将 `.env.example` 复制为本机 `.env`（已有文件不要覆盖），填写 `FINANCIAL_RAG_BASE_URL`、`NEW_API_KEY` 和 `DASHSCOPE_API_KEY`。网关地址包含 API 前缀，例如 `https://your-gateway.example/v1`。模型名称与实验参数仍由 `config.toml` 管理。

使用自己的文档需要先完成事实清洗和建库；复核历史实验需要另行取得匹配的原文、题集、运行库及复用材料。两种路线的输入要求见 [运行说明](OPERATIONS.md#输入准备与适用范围)，不能将通用文档处理能力理解为一条命令即可复现本次历史成绩。

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
