**Build your own RAG. Do not just copy it.**

# Build Your Own RAG

Build Your Own RAG 是一个包含个人偏好的 RAG 教程。它面向有编程基础的读者，主要采用 Python 原生开发，从真实场景出发，讨论解决问题时的技术选择与权衡。我希望在回答“怎么做”的同时，也讲清楚“为什么这样做”。

[项目理念](#项目理念) · [适合谁阅读](#适合谁阅读) · [整体内容](#整体内容) · [主线教程](#主线教程目录) · [特别加更](#特别加更目录) · [示例](#示例目录) · [碎碎念](#碎碎念) · [参与贡献](#参与贡献) · [许可](#许可)

## 项目理念

这个项目希望解释 RAG 中的每一步为什么存在、解决什么问题，以及如何通过实践验证它的效果。我们会讨论为什么需要 RAG、哪些场景适合采用 RAG，以及如何根据具体需求做好一个 RAG 系统。教程会覆盖从数据清洗到评测的完整流程，并按需补充工程实现方面的知识。[特别加更](#特别加更目录)则用来聊一些独立话题。

这是一个以项目和问题为导向的教程，技术服务于问题本身，每项技术选择都会放到具体场景中解释，必要时也会对比其他方案。我的个人习惯仍然会影响部分选型，读者可以参考这些选择，再根据自己的需求作出判断。

本教程不会逐行解释代码，也不会穷举运行步骤和可能遇到的报错。我认为，读者已经可以借助编程智能体（Coding Agent）处理这些实现细节。我会写清楚要做什么、解释原理，并提供相关的示例。读者需要自己动手实现和验证，理解每一步的作用。

这个项目会尽量少依赖现成的 RAG 框架或系统，例如 [LangChain](https://github.com/langchain-ai/langchain)、[LlamaIndex](https://github.com/run-llama/llama_index)、[RAGFlow](https://github.com/infiniflow/ragflow)、[LightRAG](https://github.com/HKUDS/LightRAG) 和 [RAG-Anything](https://github.com/HKUDS/RAG-Anything)。我也不是要开发一个新的 RAG 框架，而是通过示例展示如何针对具体问题实现各个环节。示例代码主要采用 Python 原生开发，并引入我认为必要的依赖。在我看来，RAG 的基本思路并不复杂，亲手实现各个环节也有助于将来在生产环境中调整和维护系统。有了 Coding Agent，在引入一个可能变化频繁的框架和自己多写一些代码之间，我会更倾向于后者。

教程以实践为主，不会从 RAG 的历史讲起，也不会展开模型训练原理。如果你希望先系统入门，或参考现有代码，可以看看 [All-in-RAG](https://github.com/datawhalechina/all-in-rag) 和 [RAG from Scratch](https://github.com/langchain-ai/rag-from-scratch)。想进一步了解不同的 RAG 技巧，可以看看按专题组织的 [RAG Techniques](https://github.com/NirDiamant/RAG_Techniques)。企业场景还会涉及权限管理、数据源连接等问题，对这些话题感兴趣，可以参考 [Azure Search OpenAI Demo](https://github.com/Azure-Samples/azure-search-openai-demo) 和 [Onyx](https://github.com/onyx-dot-app/onyx)。如果你想直接使用端到端框架快速搭建 Demo，[RAG-Anything](https://github.com/HKUDS/RAG-Anything) 和 [LightRAG](https://github.com/HKUDS/LightRAG) 都是不错的选择。

**Build your own RAG. Do not just copy it.**

## 适合谁阅读

面向有编程基础、希望理解并动手构建 RAG 应用的读者。示例主要使用 Python，建议能够阅读基础 Python 代码、安装依赖并使用命令行，同时了解如何借助 Coding Agent 处理开发问题。

如果你的主要开发语言不是 Python，也可以阅读这套教程。教程正文主要讨论通用的技术问题与解决方案，用短 JSON、文档树等示意解释数据结构；完整实现放在第六章末尾的示例中。你可以把示例交给 Coding Agent，辅助改写成自己使用的语言，再动手验证。每章需要的额外知识会在正文开头说明。

## 整体内容

- **[主线教程](#主线教程目录)**：按学习顺序展开，从真实问题出发，讨论从数据清洗到评测的方案与权衡。
- **[特别加更](#特别加更目录)**：独立讨论某个问题、实验或补充话题，与主线无关。
- **[成品示例](examples/)**：示例目录用于存放与教程相关的完整项目，每个项目独立说明环境、依赖及所用数据的获取方式。

## 主线教程目录

主线从数据清洗起步，逐步扩展到检索、生成，以及那些相对新颖和有趣的技术方案，比如结构化检索、知识图谱和类似 LLM Wiki 的知识系统。我们会在数据清洗和存储上花费不少笔墨，这些内容都是后文的伏笔。

- [001 · 什么样的数据格式，适合成为 RAG 的事实层？](tutorials/main/001-document-format.md)
- [002 · 从不同文件格式到 DoclingDocument](tutorials/main/002-docling-document.md)
- [003 · OCR 与文档解析：一些选择与取舍](tutorials/main/003-ocr-and-parsing.md)
- [004 · Chunk 与增强层：为检索准备资料](tutorials/main/004-chunks-and-enrichment.md)
- [005 · 从召回到回答：查询流程与 Agentic RAG](tutorials/main/005-retrieval-and-agentic-rag.md)
- [006 · 数据库存储与结构化回查](tutorials/main/006-databases-and-source-lookup.md)

## 特别加更目录

- [ex001 · Embedding Atlas：用可视化理解 RAG 的嵌入空间](tutorials/extra/ex001-embedding-atlas.md)

## 示例目录

示例围绕真实问题提供完整实现，供读者配合教程阅读、运行和改写。每个示例项目都使用了独立的虚拟环境，使用 UV 管理，注意阅读对应仓库文档。
- [金融长文档 RAG](examples/financial-document-rag/README.md)：对应主线前六章，以 [ViDoRe V3 Finance - EN](https://huggingface.co/datasets/vidore/vidore_v3_finance_en) 的金融财报为数据，展示从文档表示、解析、切片到检索、回读与存储的实现。

## 碎碎念

我在刚开始研究 AI Agent 的时候，对 RAG 这项技术并不热衷。少样本提示最初是有价值的，但当模型的知识储备更丰富、与人类意图的对齐更充分时，这些提示就可能从引导模型解决问题的工具，变成限制模型发挥能力的约束。我当时认为，未来是 RAG-free 的，我们应该去研究真实问题的结构，以及针对问题本身设计出尽可能通用的 Agent 结构。

不过，最近接触到更多真实生产环境中的问题并与他人交流后，我开始感觉 RAG 本身可能值得单独研究。更进一步讲，LLM Wiki 等项目让我们开始思考如何构建知识系统，而传统 RAG 是知识系统中不可缺少的一环。训练可以让模型获得知识、与人类意图对齐，却无法让它掌握现实世界中繁杂的习惯。因此，如何收集和清洗数据、如何存储知识、如何迭代知识系统，都需要单独处理。我们可能还在做一个 QA Bot，但目标是构建知识系统，让 AI 更好地理解繁杂的现实世界。

这也是整个项目的出发点，我希望能够通过一些 Case 以及对应的文档，从现实世界中存在的问题开始，思考 RAG 系统本身应该如何构建；我们又应该如何充分的利用已有的项目，来更快的做出可用的系统。

## 参与贡献

欢迎通过 Issue 或 Pull Request 纠错、补充说明、贡献示例，或推荐开源数据集。

开始之前请阅读[贡献指南](CONTRIBUTING.md)和[社区行为准则](CODE_OF_CONDUCT.md)。章节命名、图片组织与导航规则见[教程文件与导航](CONTRIBUTING.md#教程文件与导航)。

## 许可

原创代码采用 [MIT](LICENSE)；原创文档和插图采用 [CC BY 4.0](LICENSE-DOCS)。第三方内容及数据遵循各自明确声明的协议。

具体范围、署名方式和例外说明见[授权说明](LICENSING.md)。
