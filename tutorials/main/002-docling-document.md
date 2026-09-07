# 002 · 从不同文件格式到 DoclingDocument

简体中文 | 英文译文待补充

上一章：[什么样的数据格式，适合成为 RAG 的事实层？](001-document-format.md) · [返回总目录](../../README.md#主线教程目录) · 下一章：尚未发布

## 不同文件，最后放进什么样的对象

在我做过的内部文档库项目里，输入同时包含 `.docx`、`.pdf`、`.md`、`.html`、`.pptx` 和 `.xlsx`。有些内容按章节展开，有些按页面摆放，还有些信息藏在工作表和单元格之间。处理这些资料时，我希望用一种共同的表示保存内容，同时留下各自有用的结构。

Docling 是文档解析工具，`DoclingDocument` 是它提供的统一文档对象，JSON 是这个对象的一种序列化形式。我们可以使用 Docling 的解析流程，也可以用其他解析器提取内容，再通过适配逻辑构建这个对象。这里先把它的基本形式讲清楚，再介绍文档树、PPT 和表格三类组织方式。

## DoclingDocument 的基本形式

一份 `DoclingDocument` 通常包含名称、来源、Schema 版本和页面信息。其内部的内容与结构分开存放，大致如下：

```text
DoclingDocument
│
├── body: GroupItem
│   └── children
│       ├── Ref("#/texts/0")
│       ├── Ref("#/groups/0")
│       └── Ref("#/tables/0")
│
├── groups[]
│   └── GroupItem(...)
│
├── texts[]
│   ├── TitleItem(...)
│   ├── SectionHeaderItem(...)
│   └── TextItem(...)
│
├── tables[]
│   └── TableItem(...)
│
├── pictures[]
│   └── PictureItem(...)
│
├── key_value_items[]
├── form_items[]
├── field_regions[]
├── field_items[]
│
└── pages{}
```

`texts` 保存标题、段落、列表项、代码、公式和图注等文本节点；`tables` 保存表格节点及其单元格结构；`pictures` 保存图片节点，可以关联图片资源与图注。`groups` 保存容器节点，可以表示章节、列表，也能用于组织幻灯片和 Sheet。图中的键值对、表单和字段集合，先了解它们的用途即可，本章不逐项展开。

`body` 是正文树的根。它的 `children` 保存 JSON Pointer 引用，例如 `#/texts/0` 就是当前对象里 `texts` 数组的第一个元素。节点用 `self_ref` 标记自己的地址，用 `parent` 指回父节点，也可以继续用 `children` 引用子节点。沿着这些引用和子节点顺序，就能恢复正文的层级与阅读顺序。

下面用一份有标题、章节、段落和表格的文档说明。它与上图分别展示整体布局和具体引用关系，不要求节点数量相同。这个 JSON 省略了部分字段，表格内容也暂留为空：

```json
{
  "schema_name": "DoclingDocument",
  "version": "1.0.0",
  "name": "example.docx",
  "origin": {
    "filename": "example.docx",
    "mimetype": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
  },
  "body": {
    "self_ref": "#/body",
    "children": [
      { "$ref": "#/texts/0" },
      { "$ref": "#/groups/0" }
    ]
  },
  "groups": [
    {
      "self_ref": "#/groups/0",
      "label": "section",
      "name": "第一章",
      "parent": { "$ref": "#/body" },
      "children": [
        { "$ref": "#/texts/1" },
        { "$ref": "#/tables/0" }
      ]
    }
  ],
  "texts": [
    {
      "self_ref": "#/texts/0",
      "label": "title",
      "text": "文档标题",
      "parent": { "$ref": "#/body" }
    },
    {
      "self_ref": "#/texts/1",
      "label": "paragraph",
      "text": "第一章中的一段正文",
      "parent": { "$ref": "#/groups/0" }
    }
  ],
  "tables": [
    {
      "self_ref": "#/tables/0",
      "label": "table",
      "parent": { "$ref": "#/groups/0" },
      "data": { "table_cells": [] }
    }
  ],
  "pictures": []
}
```

读这份文档时，先从 `body.children` 找到标题，再进入“第一章”，依次读到段落和表格。段落的文字仍然只放在 `texts[1]` 中，章节里保存的是引用，不需要再复制正文。读取表格的 `parent`，也能知道它属于哪一章。这里用 group 表示章节只是一个例子，标题节点也可以直接作为其他元素的父节点。`self_ref` 是当前对象内的地址，可用于节点定位与索引。

`origin` 记录来源，`pages` 保存页面信息，节点的 `prov` 可以记录页码和边界框（`bbox`）。对象还有 `furniture` 根，用来组织页眉页脚等非正文内容。图片节点可通过 `image` 关联资源，通过 `captions` 引用图注；保存了节点或图片地址，并不等于已经提取到了实际图像。

在标准 Docling 流程中，解析器负责把源文件转换成这个对象，之后可以继续导出为 Markdown、HTML 或 JSON，或者在它上面做切分、摘要增强和检索。

## 文档树：DOCX、PDF、Markdown 和 HTML

这四类格式都可能包含章节、段落、列表、表格和图片。我会优先使用标准解析流程，再检查标题级别、阅读顺序和父子关系是否足以表达原文。按章节分组后，可以得到这样的结构：

```text
DoclingDocument
└── body
    ├── text(label="section_header")
    ├── group(label="section")
    │   ├── text(label="paragraph")
    │   ├── group(label="list")
    │   ├── table
    │   └── picture
    └── group(label="section")
        └── text(label="paragraph")
```

这是展开引用后的组织示意，后面的 PPT 和 Sheet 也采用同样的画法。实际 JSON 仍然把内容放在各自的集合中。图中章节标题与 group 并列是一种安排，也可以让标题成为内容的父节点；具体要看解析结果与项目采用的整理规则。

DOCX 可以利用标题样式，Markdown 可以利用标题标记。HTML 则应先确定正文范围，再利用标题、列表和表格等语义结构，避免把导航栏等页面内容混入正文。

PDF 的结构通常需要从版面恢复。文本型 PDF 即使能提取文字，也要检查多栏阅读顺序、标题层级和跨页表格；扫描页面还需要 OCR。有页码和坐标时应当保留，方便回到原页。DOCX、HTML 和 Markdown 更接近流式文档，没有可靠分页时，至少留下来源与节点引用，章节路径则根据已有层级获得。

## PPT：PPTX 按幻灯片分组

PPTX 的文字、图片和表格通常是按页面摆放的元素，页面之间未必有可靠的章节层级。我建议先让每张幻灯片对应一个 group，把同一页的元素放在一起：

```text
DoclingDocument
└── body
    ├── group(label="slide", name="slide-0")
    │   ├── text
    │   ├── picture
    │   └── table
    ├── group(label="slide", name="slide-1")
    │   └── picture
    └── group(label="slide", name="slide-2")
        ├── text
        └── table
```

这样，只有图片的页面也有自己的位置，不会因为缺少文字而从结构里消失。幻灯片标题、页面编号，以及解析时能够获得的元素坐标和来源信息，都应该保留。`slide-0` 只是这里给组起的名字，不规定源文件页码必须从零开始。

有些 PPTX 确实有目录和章节关系，可以加以利用，但我遇到的基本都没有。处理方式要根据数据本身调整，后面我们还会遇到类似的问题。

## 表格：XLSX 按 Sheet 分组

XLSX 的信息主要在工作表和单元格中。对于排班表、排期表和字段表，我建议一个 Sheet 对应一个 group，在组内保存一个或多个 table：

```text
DoclingDocument
└── body
    ├── group(label="sheet", name="目录")
    │   └── table
    ├── group(label="sheet", name="排班表")
    │   └── table
    └── group(label="sheet", name="字段说明")
        └── table
```

一个 Sheet 未必就是一张规整表格。如果里面并排放着几块表格，或者夹有说明文字，就应按实际区域组织，文字仍可作为文本节点保留。

表格内部可以继续查看 `tables[].data.table_cells[]`。以排班表中的一个单元格为例，除文字外，还要保存行列位置、合并范围和表头标记：

```json
{
  "row_span": 1,
  "col_span": 1,
  "start_row_offset_idx": 4,
  "end_row_offset_idx": 5,
  "start_col_offset_idx": 2,
  "end_col_offset_idx": 3,
  "text": "夜班",
  "column_header": false,
  "row_header": false,
  "row_section": false
}
```

这个单元格位于表格内从零计数的第 4 行、第 2 列，结束索引不包含在范围内，占一行一列。`column_header`、`row_header` 和 `row_section` 分别标记列表头、行表头和行分区。遇到合并单元格，需要保留对应的跨度与起止位置，才能知道文字覆盖了哪些行列。

CSV 也可以沿用表格的组织思路，只是没有工作簿和多 Sheet 层级。这种表示适合保留文档中的表格结构。如果面对的是需要频繁筛选、聚合或关联查询的大规模业务数据表，我仍建议使用数据库存储和查询。

## 统一表示带来系统的解耦

经过上述处理，不同格式的输入都有了统一的结构化表示。在这套组织方案中，DoclingDocument 承担结构化事实层的职责。本章讨论的格式转换只负责忠实记录源文件的内容与结构，不在这个过程中加入模型猜测。

切分、图片描述、表格摘要、Embedding、Rerank，以及 Agent 的查询和回答，都属于事实层之上的后续处理。派生内容和检索结果可以通过节点引用回到原文，查询时也可以据此扩展上下文，但这些过程不应反向修改事实层。这样做有一个实际好处：换 Embedding 模型时不需要重新解析原始文件，修改图片摘要提示词时也不会改变文档本身。

把事实层与后续处理分开，可以降低系统的耦合程度。本章先讲清楚事实层如何组织内容；后面聊完切片以后，我们会继续讨论如何利用数据库提高查询效率。

## 参考资料与署名

- Docling：[文档对象说明](https://docling-project.github.io/docling/concepts/docling_document/)与[支持格式](https://docling-project.github.io/docling/usage/supported_formats/)，用于核对对象概念与输入范围；在线文档会随版本更新。

本章原创正文与结构示意采用 [CC BY 4.0](../../LICENSE-DOCS)。
