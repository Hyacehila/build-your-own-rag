# Build Your Own RAG

[简体中文](README.md) | English

Build Your Own RAG is a tutorial shaped by my own approach to building RAG systems. It is for readers with programming experience and favors direct implementation. Starting from real scenarios, it explores the technical choices and tradeoffs involved in solving a problem. I want to explain both how to build something and why to build it that way.

[Project philosophy](#project-philosophy) · [Who this is for](#who-this-is-for) · [Content overview](#content-overview) · [Main tutorials](#main-tutorials) · [Extra chapters](#extra-chapters) · [Examples and data](#examples-and-data) · [Contributing](#contributing) · [License](#license)

## Project philosophy

This project aims to explain why each part of a RAG system exists, what problem it solves, and how to evaluate it in practice. We will discuss why RAG is needed, when it is appropriate, and how to build a system for a particular set of requirements. The tutorials will cover the process from data cleaning through evaluation, adding engineering topics where useful. [Extra chapters](#extra-chapters) make room for independent discussions, including topics outside the main sequence.

Projects and problems guide the tutorial. Each technical choice will be explained in the context of a concrete scenario, with alternatives compared where useful. My own habits will still influence some choices: I am particularly fond of [SQLite](https://www.sqlite.org/), for example. Treat those choices as references and decide what fits your own requirements.

The tutorial will not explain every line of code or catalog every setup step and possible error. I believe coding agents can already help readers handle those implementation details. I will explain what to build and the underlying principles, and provide [example code](examples/README.md) and [data](datasets/README.md). Readers are expected to implement, verify, and understand the steps themselves.

The project will keep its reliance on existing RAG frameworks and systems, such as [LangChain](https://github.com/langchain-ai/langchain), [LlamaIndex](https://github.com/run-llama/llama_index), [RAGFlow](https://github.com/infiniflow/ragflow), and [LightRAG](https://github.com/HKUDS/LightRAG), to a minimum. Examples will primarily implement the relevant parts directly, using dependencies I consider necessary. In my view, the basic idea of RAG is not complicated, and implementing its parts yourself helps when adapting and maintaining a production system. With a coding agent available, I lean toward writing more code myself when weighing that against adopting a framework that may change frequently.

The focus is practical. The tutorial will not start with the history of RAG or explain how the underlying models are trained. For a structured introduction or existing code to study, see [All-in-RAG](https://github.com/datawhalechina/all-in-rag) and [RAG from Scratch](https://github.com/langchain-ai/rag-from-scratch). For further techniques organized by topic, see [RAG Techniques](https://github.com/NirDiamant/RAG_Techniques). Enterprise use also raises questions about permissions and data-source connections; [Azure Search OpenAI Demo](https://github.com/Azure-Samples/azure-search-openai-demo) and [Onyx](https://github.com/onyx-dot-app/onyx) are references for exploring those topics.

**Build your own RAG. Do not just copy it.**

## Who this is for

This tutorial is for readers with programming experience who want to understand and build RAG applications. Examples will primarily use Python, so being able to read basic Python code, install dependencies, and use a terminal is helpful. You should also know how to use a coding agent to work through development problems.

You can follow the tutorial even if Python is not your main language. Tutorial prose discusses technical problems and solutions that apply across languages, using short JSON excerpts or document trees where they help explain data structures. Implementation code and run commands belong in the [example projects](examples/README.md). You can ask a coding agent to help adapt an example to your preferred language, then verify it yourself.

Each chapter will state any additional prerequisites. Most can be learned as you encounter them. If a concept is unfamiliar, you can also ask ChatGPT questions as you read to help you understand it.

## Content overview

- **[Main tutorials](#main-tutorials)** follow a learning sequence, using real problems to explore solutions and tradeoffs from data cleaning through evaluation.
- **[Extra chapters](#extra-chapters)** explore individual questions, experiments, or supplementary topics, including subjects unrelated to the main sequence.
- **[Complete examples](examples/README.md)** provide reference implementations cited by the chapters, each with its own environment and dependencies.
- **[Datasets](datasets/README.md)** include open datasets and data personally provided or organized by the author. Small, complete datasets that meet the distribution requirements will be included in the repository.

Chinese is the primary writing language. English translations are added chapter by chapter. Translated articles provide language-switch links. This index will include every published chapter, linking to Chinese articles with a **Chinese only** label when no translation exists. See the [translation conventions](CONTRIBUTING.md#双语支持) for details (in Chinese).

## Main tutorials

The main sequence starts with data cleaning. Chapter one first considers its desired output: a document format that preserves sources and structure for later chunking, indexing, and retrieval.

- [001 · What makes a good document format for a RAG fact layer?](tutorials/main/001-document-format.md) — **Chinese only**

## Extra chapters

None published yet.

## Examples and data

- [Complete examples](examples/README.md): the [structured document facts example](examples/structured-document-facts/README.md) demonstrates Markdown conversion, structure preservation, and version-bound node reads (guide in Chinese).
- [Datasets](datasets/README.md): no standalone datasets published yet; the example includes an original fictional teaching fixture.

## Contributing

Corrections, explanations, translations, examples, and datasets are welcome. Issues and pull requests may be written in Chinese or English.

Read the [contribution guide](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md) before contributing. See [tutorial files and navigation](CONTRIBUTING.md#教程文件与导航) for chapter naming, image organization, and navigation rules. These supporting guides are currently in Chinese.

## License

Original code is licensed under the [MIT License](LICENSE). Original documentation, translations, and illustrations are licensed under [CC BY 4.0](LICENSE-DOCS). Third-party materials and datasets follow their respective declared licenses.

See the [licensing guide](LICENSING.md) for scope, attribution, and exceptions; it includes an English summary.
