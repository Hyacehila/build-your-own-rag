# Build Your Own RAG

简体中文 | [English](README.en.md)

Build Your Own RAG 是一个带有鲜明个人风格的 RAG 教程。它面向有编程基础的读者，主要采用原生开发，从真实场景出发，讨论解决问题时的技术选择与权衡。我希望在回答“怎么做”的同时，也讲清楚“为什么这样做”。

[项目理念](#项目理念) · [适合谁阅读](#适合谁阅读) · [整体内容](#整体内容) · [主线教程](#主线教程目录) · [特别加更](#特别加更目录) · [示例与数据](#示例与数据) · [参与贡献](#参与贡献) · [许可](#许可)

## 项目理念

这个项目希望解释 RAG 中的每一步为什么存在、解决什么问题，以及如何通过实践验证它的效果。我们会讨论为什么需要 RAG、哪些场景适合采用 RAG，以及如何根据具体需求做好一个 RAG 系统。教程会覆盖从数据清洗到评测的完整流程，并按需补充工程实现方面的知识。[特别加更](#特别加更目录)（Extra Chapters）则用来聊一些独立话题，也可能涉及与主线无关的题外话。

这是一个以项目和问题为导向的教程，技术服务于问题本身。每项技术选择都会放到具体场景中解释，必要时也会对比其他方案。我的个人习惯仍然会影响部分选型，比如我非常喜欢 [SQLite](https://www.sqlite.org/)。读者可以参考这些选择，再根据自己的需求作出判断。

本教程不会逐行解释代码，也不会穷举运行步骤和可能遇到的报错。我认为，读者已经可以借助编程智能体（Coding Agent）处理这些实现细节。我会写清楚要做什么、解释原理，并提供[示例代码](examples/README.md)和[数据](datasets/README.md)供参考；读者需要自己动手实现和验证，理解每一步的作用。

这个项目会尽量少依赖现成的 RAG 框架或系统，例如 [LangChain](https://github.com/langchain-ai/langchain)、[LlamaIndex](https://github.com/run-llama/llama_index)、[RAGFlow](https://github.com/infiniflow/ragflow) 和 [LightRAG](https://github.com/HKUDS/LightRAG)。示例代码主要采用原生开发，并引入我认为必要的依赖。在我看来，RAG 的基本思路并不复杂，亲手实现各个环节也有助于将来在生产环境中调整和维护系统。有了 Coding Agent，在引入一个可能变化频繁的框架和自己多写一些代码之间，我会更倾向于后者。

教程以实践为主，不会从 RAG 的历史讲起，也不会展开模型训练原理。如果你希望先系统入门，或参考现有代码，可以看看 [All-in-RAG](https://github.com/datawhalechina/all-in-rag) 和 [RAG from Scratch](https://github.com/langchain-ai/rag-from-scratch)。想进一步了解不同的 RAG 技巧，可以看看按专题组织的 [RAG Techniques](https://github.com/NirDiamant/RAG_Techniques)。企业场景还会涉及权限管理、数据源连接等问题，对这些话题感兴趣，可以参考 [Azure Search OpenAI Demo](https://github.com/Azure-Samples/azure-search-openai-demo) 和 [Onyx](https://github.com/onyx-dot-app/onyx)。

**Build your own RAG. Do not just copy it.**

## 适合谁阅读

面向有编程基础、希望理解并动手构建 RAG 应用的读者。示例主要使用 Python，建议能够阅读基础 Python 代码、安装依赖并使用命令行，同时了解如何借助 Coding Agent 处理开发问题。

如果你的主要开发语言不是 Python，也可以阅读这套教程。教程正文主要讨论通用的技术问题与解决方案，可以用短 JSON、文档树等示意解释数据结构；具体实现代码和运行命令放在[示例项目](examples/README.md)中。你可以把示例交给 Coding Agent，辅助改写成自己使用的语言，再动手验证。

每章需要的额外知识会在正文开头说明。多数知识可以在读到时再补充；遇到不熟悉的概念，也可以边读边向 ChatGPT 提问，帮助自己理解。

## 整体内容

- **[主线教程](#主线教程目录)**：按学习顺序展开，从真实问题出发，讨论从数据清洗到评测的方案与权衡。
- **[特别加更](#特别加更目录)**：独立讨论某个问题、实验或补充话题，也可以与主线无关。
- **[成品示例](examples/README.md)**：各章节引用的项目代码参考，便于理解实现，独立说明环境与依赖。
- **[数据集](datasets/README.md)**：包括开源数据集和作者个人提供或整理的数据，符合分发条件的小型完整数据集随仓库提供。

中文是主要写作语言，英文文章按章补译。已翻译的正文提供语言切换；[英文目录](README.en.md#main-tutorials)保留全部已发布章节，未翻译的条目标注 `Chinese only`。具体规则见[双语支持约定](CONTRIBUTING.md#双语支持)。

## 主线教程目录

主线从数据清洗起步。第一章先讨论清洗的目标产物：什么样的数据格式，能够保存来源与结构，供后续切片、索引和检索复用。

- [001 · 什么样的数据格式，适合成为 RAG 的事实层？](tutorials/main/001-document-format.md)

## 特别加更目录

尚未发布。

## 示例与数据

- [成品示例](examples/README.md)：已提供[结构化文档事实层示例](examples/structured-document-facts/README.md)，演示 Markdown 转换、结构保存与节点回读。
- [数据集](datasets/README.md)：尚未发布独立数据集；示例自带原创虚构教学夹具。

## 参与贡献

欢迎纠错、补充说明、翻译文章，以及贡献示例和数据集。中文和英文的 Issue、Pull Request 均可。

开始之前请阅读[贡献指南](CONTRIBUTING.md)和[社区行为准则](CODE_OF_CONDUCT.md)。章节命名、图片组织与导航规则见[教程文件与导航](CONTRIBUTING.md#教程文件与导航)。

## 许可

原创代码采用 [MIT](LICENSE)；原创文档、译文和插图采用 [CC BY 4.0](LICENSE-DOCS)。第三方内容及数据遵循各自明确声明的协议。

具体范围、署名方式和例外说明见[授权说明](LICENSING.md)。
