# 003 · OCR 与文档解析：一些选择与取舍

简体中文 | 英文译文待补充

上一章：[从不同文件格式到 DoclingDocument](002-docling-document.md) · [返回总目录](../../README.md#主线教程目录) · 下一章：[Chunk 与增强层：为检索准备资料](004-chunks-and-enrichment.md)

## 关于 OCR，先读这篇文章

对于市面上常见的 OCR 和文档处理方式，我已经单独写过一篇[《让 OCR 再次伟大》](https://hyacehila.github.io/blog/2026/07/04/make-ocr-great-again/)，这里把它作为核心阅读内容，不再完整展开。本章补充两个实际问题：面对自己的文档应该先检查什么，以及为什么在这套教程里，我选择了 Docling，而没有考虑使用 Unstructured。

## 先判断输入，再决定怎么解析

第二章已经说明我们希望得到怎样的文档对象。这一步要考虑的是，怎样从源文件中获得它。能够提取文字，不代表阅读顺序、标题关系和表格已经正确；需要 OCR，也不代表所有页面都必须采用同一种处理方式。

| 输入情况 | 可以先尝试什么 | 对照原文检查什么 |
|---|---|---|
| 文字层可用、版面常规的 PDF | 原生文字提取与文档解析 | 文字是否完整，段落与标题是否对应，表格行列是否正确 |
| 扫描页或文字层损坏的页面 | 适用语言的 OCR，再恢复版面与结构 | 缺字、乱码、数值、阅读顺序和来源位置 |
| 多栏、跨页表格或复杂图文 | 抽取代表页，比较解析配置；必要时引入其他解析器的候选 | 跨栏混排、续表关系、表头、图注与坐标映射 |
| 文字正确，但标题层级或归属错误 | 检查结构整理规则与标题证据 | 父子关系、章节边界和附件范围；不要为此重识别正文 |

这些是选择起点，不是“某一工具适合所有文档”的结论。我会先找几页有代表性的材料，对照原页看结果，记录失败位置，再决定是否调整配置或换一条解析路线。只看到解析成功状态，或者导出的 Markdown 看起来很顺，还不足以判断事实层已经可用。

## 用金融示例检查交付结果

[金融教学路线](../../examples/financial-document-rag/TEACHING.md)提供原始 PDF、已知布局的 DoclingDocument 和后续回读入口。先对照它们，确认自己理解章节、表格、图片和来源位置怎样表达。这个教学对象是按布局构造的，不能用它验证 OCR 或真实版面解析效果。

验证自己的解析路线时，选一份有权使用的小型文档，按[文档解析练习](../../examples/financial-document-rag/TEACHING.md#文档解析的练习)准备独立输入。至少检查一个正文区域、一张表或图片、一个标题边界，确认它们能回到正确的原页。发现问题时，区分文字识别错误、版面顺序错误、结构关系错误和来源映射错误；这些问题需要的修复方式不同。

## 为什么我没有选择 Unstructured

Docling 和 Unstructured 都是不错的文档处理工具，可以把 PDF、DOCX、PPTX、XLSX、HTML 和图片等输入转换为各自的结构化文档表示，也都提供切片能力。

我对 Unstructured 的解析质量仍有顾虑，尤其是在选择本地开源方案时。它的[官方说明](https://docs.unstructured.io/open-source/introduction/overview)也列出了开源库相对其商业服务在文档与表格提取、层级识别方面的限制。这些说明不能直接证明它不如 Docling。就我目前的选型判断，在解析质量和 OCR、VLM 后端的选择上，我更倾向于 Docling，也更关心一个实际问题：如果默认的处理效果不够好，我能怎样继续调整？

我更看重 Docling 提供的统一文档表示，以及围绕这个表示替换处理组件的能力。Docling 本身也提供解析流程，但[后端与 pipeline 可以配置和扩展](https://docling-project.github.io/docling/concepts/architecture/)，部分处理还可以[接入远程模型服务](https://docling-project.github.io/docling/usage/advanced_options/#using-remote-services)。对我来说，这意味着可以先确定最终要保留的文档对象，再根据输入选择合适的处理方式。对于没有现成接口的服务，也可以编写适配逻辑，把它的输出整理成 DoclingDocument。

例如，某类 PDF 的处理效果不好时，我希望能够调整 OCR、解析后端或模型服务，再把结果整理进 DoclingDocument，让后续的切片和查询继续使用统一对象。外部解析结果往往仍需适配，节点类型、阅读顺序和来源位置都需要对应起来。我愿意做这部分工作，以便根据数据调整具体组件。我在博客里就聊了如何结合 MinerU 和 Docling，实现更好的效果。

两者在文档模型上的组织方式也不同。Unstructured 的文档处理接口以元素为中心，可以通过元数据表达层级关系；DoclingDocument 则按类型保存节点，再通过引用组织文档树，后者对我更直观。在切片时，Unstructured 也会利用文档元素，并支持按标题保留章节边界和类似 LangChain 的各种切片工具。我更喜欢 Docling 围绕文档对象进行[结构感知切片](https://docling-project.github.io/docling/concepts/chunking/)，再通过 `contextualize()` 把标题等上下文补入切片文本的方式，这与我保留统一事实层的思路更契合。

Unstructured 还有一个值得考虑的优势：数据接入（Ingestion）。它的 Ingest 工具按 source connector → processing → destination connector 组织流程，提供 [Google Drive](https://docs.unstructured.io/open-source/ingestion/source-connectors/google-drive)、[GitHub](https://docs.unstructured.io/open-source/ingestion/source-connectors/github) 等现成的数据源连接器。如果企业知识来自这些服务，Unstructured 可以减少接入工作。相比之下，Docling 核心更侧重文档转换，数据源采集和同步通常需要配合其他工具完成。

我喜欢 DoclingDocument 的表示方式，这是我选择它的核心原因。不过，如果你的主要需求是接入多个数据源，Unstructured 的连接器也值得考虑，具体选择还是要看自己的数据和处理流程。

本章原创正文采用 [CC BY 4.0](../../LICENSE-DOCS)。
