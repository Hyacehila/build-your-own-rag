# 金融长文档 RAG 实验（开发中）

[示例目录](../README.md) · [设计](DESIGN.md) · [操作](OPERATIONS.md) · [当前验收](VALIDATION.md) · [实验交接](EXPERIMENT_HANDOFF.md) · [代码导航](DEVELOPMENT.md)

本示例以六份 ViDoRe V3 Finance 财报为材料，比较从 PDF 原生文字到结构化事实、图表增强和可读原文的 Agent 检索。当前 **v2 事实发布**及真实 A/B 检索层可离线恢复、检查。状态数字只在[当前验收](VALIDATION.md)维护。

## 四组实验

| 组别 | 输入与检索 | 答案路径 |
|---|---|---|
| A | PDF 原生文字的平面切片，稀疏与向量混合检索 | Direct：检索、回读、回答 |
| B | 已审阅 DoclingDocument 的结构切片，保留标题与来源 | Direct：检索、回读、回答 |
| C | 在 B 的事实基础上增加图片描述与整表摘要 | Direct：检索、回读、回答 |
| D | 复用 C 的增强索引 | Agentic：模型调用 `search`/`read` 再回答 |

## 当前进度

- **已完成：**六份财报的 v2 事实发布、标题树审阅与来源链；真实财报的 A/B 切片、向量、混合检索及异地恢复；虚构小样的 A/B/C/D 与真实 v2 A/B 的有限 API 链路预实验。
- **进行中：**针对已知解析缺陷的定点候选复核。Wells Fargo 物理第 335 页的 MinerU 改善只作为候选，未并入 v2 发布。
- **未完成：**真实六份财报的完整 C/D 增强索引、正式题集与页级标注绑定、309 题正式四组实验和答案质量评测。现有链路通过不代表效果已验证。

## 阅读顺序

先读[设计](DESIGN.md)理解事实版本与四组差异，再看[当前验收](VALIDATION.md)确认哪些结论已有证据。需要恢复快照、查看 Docling Studio 或继续实验时用[操作手册](OPERATIONS.md)。事实清洗的内部执行规范在[项目 Skill](.agents/skills/clean-financial-documents/SKILL.md)。

正式实验的共享配置是[config.toml](config.toml)；[config.offline.toml](config.offline.toml) 仅用于虚构夹具的本地检查。密钥只放环境变量或被忽略的 `.env`。原始 PDF 和工作缓存不随仓库发布。两份不同用途的 v2 SQLite 用 Git LFS 交接。
