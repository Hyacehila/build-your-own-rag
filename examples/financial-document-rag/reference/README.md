# 金融教学参考材料

[返回教学路线](../TEACHING.md) · [返回示例说明](../README.md)

这里提供本项目原创的三页虚构银行报告，供第一章及后续章节直接阅读。它们由 [fixture.py](../src/financial_rag/fixture.py) 和 [教学入口](../tools/teaching_example.py)生成，不含真实财报、ViDoRe 题集或历史实验记录。

- [fictional-bank.pdf](fictional-bank.pdf)：原始教学 PDF，包含业绩段落、收入图、业务收入表和流动性说明。
- [document.json](document.json)：按已知版面显式构造的完整 DoclingDocument，保留层级和来源位置。
- [structure.txt](structure.txt)：按文档正文树的实际顺序展开的节点引用。

结构对象不是运行神经 PDF 解析器得到的结果。它用于说明统一表示、结构化切片和原文回读，不证明 OCR、版面解析或问答效果。夹具中的公司和数值都是虚构的；`Test segment` 行用于长表拆分测试，不代表真实业务数据。

生成运行材料时，使用教学路线中的命令，新输出写入被忽略的 `.local/`。参考对象与当前生成结果通过结构和内容对照维护。

这些原创教学材料及说明采用 [CC BY 4.0](../../../LICENSE-DOCS)，生成代码采用 [MIT](../../../LICENSE)。
