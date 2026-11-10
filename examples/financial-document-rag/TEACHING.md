# 用金融文档贯穿前几章

[返回示例说明](README.md) · [主线目录](../../README.md#主线教程目录) · [直接阅读参考材料](reference/README.md)

先从一份三页的虚构银行报告理解数据流，再阅读真实财报的清洗与实验代码。教学夹具由本项目的 `fixture.py` 原创生成，银行、数值、问题和答案均为虚构。

## 先看一份文档留下了什么

不安装依赖也可以对照[原始 PDF](reference/fictional-bank.pdf)、[DoclingDocument](reference/document.json)和[文档树](reference/structure.txt)。第一页是业绩段落和收入图，第二页是业务收入表，第三页是风险与流动性说明。

这份结构对象按已知版面显式构造，页码与区域来自生成 PDF 时的布局。它演示数据表示、切片和回读，不是神经 PDF 解析器的输出，也不能用于证明 OCR 或版面识别质量。表中的 `Test segment` 行专门用于测试长表拆分，不参与真实业务统计。

## 本地生成与回读

安装方式见[首次使用](README.md#首次使用)。以下命令在本例目录执行，不需要 API 密钥、历史原文库或模型权重：

```powershell
uv run --locked --offline python -B tools/teaching_example.py
```

程序使用 `config.offline.toml`，只在新目录 `.local/teaching/` 中生成夹具、读取 PDF 原生文字、构建平面与结构化切片，并按来源节点读取完整收入表。已有目录不会覆盖；再次运行时用 `--output .local/teaching-second` 指定另一个目录。

| 文件 | 阅读时关注什么 |
|---|---|
| `fixture/fictional-bank.pdf` | 原始文字、图表与物理页 |
| `document.json`、`structure.txt` | 标题层级、节点类型、阅读顺序及来源位置 |
| `chunks.json` | 平面与结构化切片的差别、标题上下文与来源引用 |
| `readback.json` | 表格原文、实际读取记录与证据的物理页绑定 |
| `experiment.sqlite` | 本次夹具的节点、切片及关系记录 |

结构化对象中的 `#/tables/0` 属于 `#/texts/3`（业务收入小节），再向上是 `#/texts/0`（业绩章节）。表格共 28 行、3 列，`2024 USD m` 列中的 `Retail` 收入为 `100`，来源为物理第 2 页。回读应保留整表及这些来源信息。

这一步没有生成摘要、向量或模型答案。后续理解完整金融 RAG 时，再沿下面的实现位置阅读；真实服务调用的输入与操作要求见[运行说明](OPERATIONS.md#输入准备与适用范围)。

## 章节与实现怎么对应

| 章节 | 先观察什么 | 继续阅读的实现 |
|---|---|---|
| [001：事实层格式](../../tutorials/main/001-document-format.md) | 表格属于哪一节，图注和图片怎样关联，页码与坐标是否存在 | [夹具构造](src/financial_rag/fixture.py)、[节点投影](src/financial_rag/parsing.py) |
| [002：统一文档对象](../../tutorials/main/002-docling-document.md) | `body`、`texts`、`tables`、`pictures` 如何通过引用组成文档 | [参考对象](reference/document.json) |
| [003：OCR 与解析](../../tutorials/main/003-ocr-and-parsing.md) | 区分原生文字提取、结构构造与真实版面解析 | [解析流程](src/financial_rag/parsing.py)、[清洗技能](.agents/skills/clean-financial-documents/SKILL.md) |
| [004：Chunk 与增强](../../tutorials/main/004-chunks-and-enrichment.md) | 切片保留了哪些来源，表格怎样拆分和重复表头 | [切片](src/financial_rag/chunking.py)、[增强](src/financial_rag/enrichment.py) |
| [005：检索与回答](../../tutorials/main/005-retrieval-and-agentic-rag.md) | 候选如何回到原文，读取结果怎样成为证据 | [检索](src/financial_rag/retrieval.py)、[来源读取](src/financial_rag/sources.py)、[五组编排](src/financial_rag/experiments/) |
| [006：存储与回查](../../tutorials/main/006-databases-and-source-lookup.md) | 文档版本、节点、切片与来源关联如何保存 | [存储](src/financial_rag/storage.py)、[结构回查](src/financial_rag/readback.py) |

章节讨论通用思路，金融实验有自己的参数和约束。例如它的 A/B/C 使用原始问题检索，D 才先重写问题；表结构也是项目实现，不要求与教程的 SQL 示意逐字段相同。对照时关注各阶段的职责和来源关系，不把某一实验的固定配置当作所有 RAG 的要求。

## 文档解析的练习

这份夹具适合检查表示和回读。需要验证真实解析时，另选一份有权使用的小型 PDF，先看原生文字是否可靠，再对照标题、表格、阅读顺序与原页位置。按[清洗技能](.agents/skills/clean-financial-documents/SKILL.md)准备输入，采用独立工作目录，不套用六份真实财报的清洗 profiles。扫描页、跨页表格和复杂版面需要各自的原页对照，不能用已知布局夹具的成功代替。
