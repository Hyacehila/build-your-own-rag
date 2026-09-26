# 代码导航与维护

[项目首页](README.md) · [设计](DESIGN.md) · [操作](OPERATIONS.md) · [实验交接](EXPERIMENT_HANDOFF.md)

## 先读实验主流程

普通实验从 `finance-rag` 进入。CLI 只处理参数、数据库生命周期、JSON 输出和退出码；数据准备、实验和诊断的参数与适配函数分别放在 `commands/` 的三个模块。算法仍在对应领域模块中，不在 CLI 内实现第二份。

```text
src/financial_rag/
  cli.py                    统一命令入口
  commands/
    data.py                 download / parse / chunk / inspect / migrate
    experiment.py           enrich / embed / run / evaluate / report / usage
    diagnostics.py          audit / repair-preview / source / fixture / 小样
  dataset.py                固定来源、英文题集和页级映射
  parsing.py, facts.py       PDF 解析、快照和事实节点
  chunking.py                A/B 及增强索引切片
  enrichment.py             图像描述、整表摘要
  retrieval.py              嵌入、稀疏/向量混合检索
  sources.py, readback.py    原文与原页回读、来源约束
  runner.py                 Direct / Agentic 逐题执行和续跑
  evaluation.py             检索指标、judge 和结果报告
  cleaning.py               清洗候选构建、事实发布/恢复
  final_outline.py          最终标题树审阅回执与验收
  retrieval_release.py      A/B 发布、恢复与向量复用
  diagnostics/              只读审计、质量诊断和修复候选
  config.py, api.py          配置契约与模型协议
  storage.py, common.py      存储、序列化、哈希
```

运行四组实验时先看 `config.py → chunking.py/enrichment.py → retrieval.py → sources.py → runner.py → evaluation.py`。恢复别人交付的索引先看 `retrieval_release.py`。修改事实清洗才需要深入 `cleaning.py`、profiles 和项目 Skill。

## 诊断与事实构建的边界

`diagnostics/` 集中了原先散在包顶层的审计、画廊、来源检查、页级清洗审计、解析质量检查、原始数据质量检查、修复预览和结构修复候选。它们有测试和实际调用，不能因为不出现在正式 run 中就删除。

`diagnostics/text_checks.py` 仅做词项、数字及覆盖率比较。解析检查和表格修复直接依赖这些纯函数，不再为几个文本比较函数加载完整审计流程。诊断指标不能被当成语义准确率或自动修改事实的依据。

以下模块继续保留：

| 内容 | 保留原因 |
|---|---|
| `mineru_cloud.py`、`mineru_packets.py`、两个 bridge | 问题页试验、批量页包和受支持格式转换仍被清洗入口调用；二者并非完全重复 |
| `table_repairs.py`、`asset_roles.py`、`outline.py`、`page_exclusions.py`、`cleaning_profiles.py` | v2 清洗与来源保持需要这些实现 |
| `facts.migrate`、legacy 配置语义 | 用于旧快照和已有缓存恢复，删除可能迫使重解析或重嵌入 |
| `fixture.py`、`validation.py`、`config.offline.toml` | 无真实数据测试，以及有边界的 API 协议验证 |
| `tools/docling-studio/` | 可选查看工具；正式实验不要求启动，但人工核查仍可使用 |
| [历史 profiles](profiles/archive/README.md) | 保留旧版本复现依据，与当前默认 v2 分开 |

`cleaning_audit.py` 顶层只保留兼容入口，原有 `python -m financial_rag.cleaning_audit` 仍可用。其实现位于 `diagnostics/cleaning_audit.py`，默认事实目录已改为当前 v2。其他诊断模块的 Python import 路径改为 `financial_rag.diagnostics.<模块名>`；本项目调用和测试已同步。正式 `finance-rag` 的 25 个命令名称、参数和只读模式保持原样。

## 配置、历史材料与提交范围

- `config.toml` 是共享实验契约；修改目录或凭据使用被忽略的 `config.local.toml`、`.env`。
- `profiles/cleaning-v2/` 是当前 profile；`profiles/archive/cleaning-v1/` 仅用于历史对照，文件内容未改。
- `.local/` 下的快照副本、尝试、备份、账本和生成报告保留在本机，不作为代码提交。不要按文件名猜测无用就删除研究记录。
- `handoff/.gitattributes` 是两个已发布 SQLite 的 LFS 规则来源；项目根目录的重复规则已移除，文件仍由 LFS 跟踪。
- 代码交付应包含源码、测试、锁文件、配置示例、profiles、项目 Skill、工具和项目文档。排除 `.env`、虚拟环境、原 PDF 与工作缓存。数据库已单独发布，代码整理不生成新的数据库版本。

本轮整理重点是边界与可读性，没有更换解析器、模型、事实清洗规则、切片算法或评分标准。原始大文件、Skill 和回归测试仍有用途，不做机械删减。还存在的全量题集接入、C/D 验收和费用限制任务见[实验交接](EXPERIMENT_HANDOFF.md)，不能把目录整理等同于全量实验就绪。

## 版本与验证

解析、切片和模型的语义签名与代码文件路径分开。只调整代码组织不要求重建已验收的事实或重新付费嵌入。运行记录的 implementation 指纹现在递归覆盖子目录，并以相对路径区分同名文件；改代码后应使用新的 run ID，不能把旧预实验记录当成同一实现续跑。

在项目目录运行：

```text
uv sync --locked
uv run --locked ruff check src tests tools
uv run --locked pytest -q -m "not docling"
uv run --locked python tools/check_docs.py
uv run --locked python -m financial_rag.retrieval_release check
```

最后一项要求本机已有恢复工作库和原 PDF；它使用缓存向量，不调用新模型。其他离线测试使用虚构数据和本地协议替身，不证明真实模型的答案质量。真实解析器测试单独标记为 `docling`，需要显式准备模型权重。

文档检查验证必需文档、链接和发布元数据，不再要求 Markdown 文件总数固定，增加维护文档无需修改一个无关的计数。正式提交前还需核对暂存文件清单与密钥排除规则。

2026-09-25 本轮整理验证：71 项离线测试通过，1 项真实解析器测试未运行；Ruff、文档检查和 wheel 打包通过。与整理前保存的基线相比，全部 CLI 参数、只读模式、解析/切片/嵌入签名、配置、锁文件和 profile 内容一致。真实 A/B 工作库检查再次通过，事实发布 ID、索引 ID 与切片数量保持不变，新增模型请求为零。剩余 Docling 弃用警告属于依赖接口迁移事项，本轮未改变解析实现来消除警告。
