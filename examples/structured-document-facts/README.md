# 结构化文档事实层：从 Markdown 到 DoclingDocument

[返回示例索引](../README.md) · [原创教学夹具](fixtures/README.md) · [参考输出](reference/README.md)

这个示例把一份虚构的产品复盘 Markdown 转成 DoclingDocument，再按原有标题级别整理层级、保存 JSON，并通过绑定版本的节点引用回读。输入、转换规则和结构化结果都可检查，不需要 Lenny Compass 的真实数据。

关联教程尚未发布。示例先提供可运行的实现和参考输出。

## 数据流与实现选择

```text
原始 Markdown 字节
  → SimplePipeline + Markdown 后端
  → 适配前的 DoclingDocument（保留作诊断参考）
  → 按已有标题级别组织层级，显式保存图片引用和图注
  → 文档 JSON + 来源清单 + 原始快照
  → 校验文档版本，按节点引用读取内容和父级路径
```

[fact_layer.py](fact_layer.py) 实现转换、适配、存储和回读，[demo.py](demo.py) 提供命令行入口。本例没有切片器、Embedding、查询服务或模型调用。

固定版本的 Markdown 后端在这份输入中保留了标题级别，但章节标题、段落和表格仍是正文根节点的直接子节点。`nest_sections` 按既有标题级别维护一个栈，将这些节点组织到对应标题下。它保留原有文字和阅读顺序，不生成新标题，也不宣称能恢复任意文档的语义结构。

图片处理也有一条显式规则：用 Markdown 图片地址创建引用，用其替代文字作为本例图注。若后端已经将相同替代文字产出为图片后的文本节点，就复用该节点并标为 `caption`，不再复制一遍。实际图片未被读取，因此图像尺寸为 `0 × 0`、DPI 为构造引用所需的占位值；这些值不代表测量结果。图片与图注用引用关联，不要求图注一定是图片的子节点。

来源清单记录这两条规则。它们是确定性适配，与模型补写标题不同。本例不执行模型增强。

## 环境与安装

- Python 3.11；已在 Windows、CPython 3.11.14 上验证。
- 使用 uv 管理独立环境，依赖固定在 [pyproject.toml](pyproject.toml) 和 [uv.lock](uv.lock)。
- 不需要 GPU、API 密钥或外部模型服务；首次安装需要下载 Python 软件包。安装后运行不需要下载模型权重。

从仓库根目录进入本例，然后安装：

```powershell
cd examples/structured-document-facts
uv sync --frozen
```

本例选择 `docling-slim[format-markdown]==2.120.1` 和 `docling-core==2.91.0`，沿用参考项目的版本。这个版本的 `DocumentConverter` 即使只处理 Markdown，也会在导入时加载 PDF 和阅读顺序相关模块，因此仍需保留若干导入依赖，安装的包中包括 PyTorch。安装这些包不代表运行了 OCR 或下载了模型权重。

## 构建与读取

以下命令都在本例目录执行：

```powershell
uv run --frozen demo.py
uv run --frozen demo.py validate
uv run --frozen demo.py read
```

第一条命令构建 `outputs/demo/`，打印结构树并回读首个表格。第二条校验快照完整性。第三条使用生成的 `locator.json` 再次读取同一节点。

已有输出目录不会被覆盖。再次构建时换一个目录：

```powershell
uv run --frozen demo.py build --output outputs/second-run
uv run --frozen demo.py read --output outputs/second-run
```

可以通过 `--source` 指定自己有权使用的 UTF-8 Markdown，通过 `--document-id` 给出本地文档标识。它是调用者管理的 ID，不是自动生成的全局身份。转换范围以本例验证的 Markdown 元素为准。

读取其他节点时，复制输出中的 `locator.json`，只将 `self_ref` 改成 `structure.txt` 中的实际引用，再通过 `--locator` 指定该文件。例如：

```powershell
uv run --frozen demo.py read --locator outputs/my-locator.json
```

该命令要求你先创建 `outputs/my-locator.json`。不要改动绑定的文档 ID 和哈希来绕过版本错误；旧 locator 应匹配旧快照，文档更新后应重新生成定位信息。

## 输出与预期结果

| 文件 | 用途 |
| --- | --- |
| `source.md` | 实际参与转换的原始字节快照 |
| `parsed-document.json` | 规则适配前的对象，供诊断比较，不作为读取入口 |
| `document.json` | 规则适配后的完整 DoclingDocument |
| `manifest.json` | 原始文件名、源文件 SHA-256、文档 ID、对象 SHA-256、构建版本、解析器版本及适配规则 |
| `locator.json` | 文档 ID、对象 SHA-256 与 `self_ref`，默认指向首个表格 |
| `structure.txt` | 对保存的 `children` 关系按顺序遍历所得的结构树 |
| `preview.md` | 面向阅读的预览；不是来源快照，也不承担完整结构存储 |

样例中两个“观察结果”应分别属于“新团队邀请实验”和“现有团队权限检查”。首个表格位于前者，保留 3 行、4 列，表头包含“调整后”，对应示例值为“55%”。这些数值全部是虚构数据。

回读结果包含节点完整字段、从根节点到父节点的路径、来源文件名与哈希。Markdown 输入不提供可信的页面坐标，本例的 `pages` 和节点 `prov` 均为空，不虚构页码、坐标或源文件行号。

`manifest.json` 最后写入，作为构建完成标记。写入中断后可能留下不完整目录，此时验证失败；重新构建应使用新目录。本例没有实现生产系统的原子代际发布。

## 验证与错误边界

```powershell
uv run --frozen pytest -q
```

2026-09-07 在 Windows、Python 3.11.14 上完成了构建、验证和回读。测试在禁止网络连接的条件下运行，检查标题归属、表格、列表、图片引用、原始字节、序列化往返和失效定位等行为，具体断言见[测试文件](tests/test_fact_layer.py)。

| 错误 | 含义 |
| --- | --- |
| `OUTPUT_EXISTS` | 输出目录已经存在，程序拒绝覆盖 |
| `SOURCE_MISMATCH` / `DOCUMENT_MISMATCH` | 快照中的源文件或文档对象与保存的哈希不一致 |
| `STALE_LOCATOR` | 定位信息属于其他文档或其他文档版本 |
| `MISSING_REF` | 节点不存在，或不能从正文树中到达 |
| `INVALID_BUNDLE` / `INVALID_TREE` | 快照缺失、格式无效或树关系不一致 |

哈希检查保证这些文件之间的一致性，不证明原文内容真实，也不是数字签名。示例只检查正文树中的结构引用，不是覆盖 Docling 所有元素和语义关系的通用文档质检器。

## 来源与简化范围

本例参考作者的私有项目 Lenny Compass，提交为 `09ba9e449712f9a1d5a899584e2949d15ef9c321`。主要参考 `pipelines/fact_build/src/fact_build/builder.py` 的 Markdown 转换、图片引用、JSON 往返及哈希逻辑，以及 `contracts/src/ai_pm_contracts/facts.py` 的版本绑定定位约定。

代码已整理为独立示例；标题层级适配是本例针对实际转换结果增加的规则。没有迁入语料目录、增强全文、chunks、向量或索引，也没有迁入源项目的批量构建、隔离区、切片、检索与回答流程。学习本例不需要访问私有仓库。

原创代码采用 [MIT](../../LICENSE)，原创说明和教学夹具采用 [CC BY 4.0](../../LICENSE-DOCS)。依赖库保留各自许可；Docling 与 Docling Core 的来源见[项目主页](https://github.com/docling-project/docling)和[核心类型库](https://github.com/docling-project/docling-core)。
