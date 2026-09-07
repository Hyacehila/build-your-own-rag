# 可直接阅读的参考输出

[返回示例说明](../README.md) · [原始教学文档](../fixtures/product-review.md)

以下文件由本例的原创虚构输入实际生成，用于在 GitHub 上阅读和核对数据结构：

- [parsed-document.json](parsed-document.json)：Markdown 后端输出，尚未应用本例的规则适配。
- [document.json](document.json)：保留图片引用及图注，并调用 Docling Core 的 `_hierarchize()` 组织章节后的完整文档对象。
- [structure.txt](structure.txt)：适配后文档的正文树，包含实际节点引用。

两份 JSON 都是 DoclingDocument。前者已有标题级别和阅读顺序，后者另外保存了显式章节父子关系；不是从无结构文本升级成另一种格式。查看两份文件中的 `#/tables/0`：其 `parent` 分别为 `#/body` 与 `#/texts/4`。处理步骤与字段对照见[示例说明](../README.md#三个阶段分别看什么)。

它们不是 Lenny Compass 的文档或其他真实语料。这一目录只保留教学参考文件，不是完整的运行快照；需要执行回读时，按示例说明构建 `outputs/` 中的完整快照。

参考输出随固定版本的代码和输入维护，测试会核对它们与重新构建的结果一致。修改输入、适配规则或解析器版本时，应重新生成参考输出并复核正文中的摘录。

这些由原创教学夹具派生的参考文件采用 [CC BY 4.0](../../../LICENSE-DOCS)。
