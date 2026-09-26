# 金融 RAG 项目适配（可选）

仅在 examples/financial-document-rag 项目执行时读本页，先核对 AGENTS.md、DESIGN.md、OPERATIONS.md 和源码。通用 Skill 交付到 DoclingDocument；这里说明本项目如何实现这些步骤。SQLite 是现有程序的工作存储，不是移植 Skill 的前提。

## 工具、步骤和实现位置

| 通用步骤 | 当前实现 | 需要注意 |
|---|---|---|
| PDF 预检与初始解析 | parsing.py、diagnostics/parsing_quality.py、PyMuPDF、Docling | 初始 CLI 依赖项目输入清单，不能直接声称支持任意 PDF 路径 |
| 导航排除与标题样式/结构重放 | page_exclusions.py、cleaning_profiles.py、outline.py | 当前清洗从既有全量快照排除导航；解析前分段排除路线需另行适配页映射 |
| 本地小样诊断 | parse-sample、parsing_guards.py | 按具体缺陷选择后端、单元格匹配或 OCR；不必重跑全本 |
| MinerU 问题页包 | mineru_cloud.py / mineru_packets.py | 两个入口配置不同，执行前核对请求 manifest 和预算 |
| 外部格式转换及资产修复 | mineru_layout_bridge.py、table_repairs.py、asset_roles.py | 固定支持的 layout schema；不是任意 MinerU 版本的通用适配器 |
| 候选 DoclingDocument 构建 | cleaning.py | 复用原始/云端缓存，保留替换和拒绝记录 |
| 最终标题包与验收 | final_outline.py | 根据最终快照审阅并签收，不由脚本代替 Agent 阅读 |
| 可选查看 | tools/docling-studio | 独立查看副本，不能以显示成功代替验收 |

使用项目 Python 3.11、uv 和锁文件。当前已验证 Docling 依赖包括 docling-slim 2.120.1、docling-core 2.91.0、docling-parse 7.13.0、docling-ibm-models 3.14.0；以运行记录为准。数字财报曾用 CPU、accurate 表格及关闭 OCR，不照搬给扫描材料。本地 OCR 入口目前写死英语，其他语言先适配，不能只改语言说明就宣称支持。

MinerU 当前封装采用精准解析 vlm、英语、文件 is_ocr=false，表格开启；公式选项在 cloud 与 packets 路线分别为关闭和开启，必须保存实际 manifest。其他输入/模式需适配配置并验证。layout bridge 使用受支持结构中的 preproc_blocks 等字段保留页内区域。

### 路由示例（来自 V1/V2 审查，不是新的项目解析实现）

- Wells Fargo 物理第 335 页：V2 的 Docling 正文存在缺字和阅读顺序混杂；既有 MinerU 单页诊断候选在原页对照中恢复了词语和段落顺序，来源映射检查通过，但候选没有合入 V2。这个例子支持“已定位的正文/版面问题 → Docling 局部候选仍失败 → MinerU 局部对照”，不能据此推断全书切换必然更好。
- Wells Fargo 物理第 112、121 页：旧版把清晰的 `I.` 字母条目误设为更高标题级别，错误祖先由标题序列/层级判断造成；V2 已修正并通过预期父节点审阅。此类问题走结构重建和标题树验收，不靠重跑解析器解决。
- 既有第 335 页试验的请求清单指定 `vlm`，服务结果元数据报告 `_backend=hybrid`、`_version_name=3.4.4`。因此，对照时要以服务实际返回的模型/后端元数据为准；请求参数不能冒充已验证的实际配置。

这些案例用于澄清 Skill 的路由边界，不改变本项目解析器、配置、云端额度或 V2 发布。

表格风险页包通常带必要邻页；当前一对一匹配门槛包括 IoU 至少 0.6、数字覆盖下降不超过 0.01。这些是实现参数，不证明错列已消失，不能推广为所有文档的质量标准。

## 输入、决策和缓存

新文档先补齐唯一 doc_id、PDF 哈希/页数、语言、原始快照及逐页来源清单。现有 prepare 从 Store 的 documents/rawparse 指针读取；build 从 workspace 的 sources.json、diagnostics、profiles 和可选 cloud 候选读取。sources.json 可含本机绝对路径，不能当通用便携交付清单。

Profile 绑定 doc_id、pdf_sha256、raw_snapshot_sha256。excluded_pages、boundaries、heading_levels、reviewed_pages 为当前构建所需，保留标题须显式分类。原始 ref 经目录排除映射后执行，可选操作包括：

- demote、furniture、navigation_refs：降级伪标题，移出重复栏头或混合页导航。
- promote：把已有完整叶文本提升为标题，使用同页 before_ref 和证据；不能编写新标题。
- nest：按级别归于同范围前置标题，并验证是否被中间标题遮蔽。
- reparent：有证据地修正错挂子树，验证同范围、无环和来源保持；不泛化为任意全文重排。
- empty_tables_as_pictures：只转换原页确认的空表图示占位；拒绝抹掉真实单元格。

每条决定记录原页证据、前后关系和理由。当前执行器只对部分操作强制证据字段，Skill 要求比现有自动校验更完整；不能声称脚本已验证每条语义决定。

当前已验收事实与最终标题回执分别保存在独立的本机工作目录；不原地重建。旧工作目录中的原始/云端材料只有在来源哈希和配置核对通过时才可作为新候选输入，目录名称本身不构成版本依据。为新候选指定独立 output 和 profile。

## 当前回执适配

final_outline 使用 contract、doc_id、parse_id、snapshot_sha256、outline_sha256 绑定候选；reviewer 与 final_tree_reviewed 标记真实审阅。reviewed_heading_refs 覆盖最终标题各一次，expected_relations 记录有原页证据的父节点或父标题关系；contents_pages_viewed、representative_pages_viewed、suspicious_pages_viewed 记录实际查看页，无目录时提供 no_contents_reason。

confirmed_wrong_parent_remaining 在通过时为空；uncertain_exceptions 明确未决原因。包哈希、分支复用依据和成本可另存，但现有校验器未全部强制检查。默认标题包 100 项，可按长度调整。

当前校验器要求代表页及至少一条预期关系，零标题材料需实现空树验收适配。通用 Skill 允许真实平坦文档；不能为满足此处程序限制造标题。Agent 阅读边界和疑点重试预算也是执行规范，尚非代码强制限流。

## 项目执行入口与下游边界

已准备输入与 profile 后，使用 cleaning build 构建候选，使用 cleaning check --no-chunks 检查事实。结合 cleaning_audit 的机器诊断，再执行 final_outline packets → 实际目录/标题审阅 → accept → verify。必须显式指定各命令的 output/root/reviews，避免默认值指向已有发布；当前操作入口见项目根目录 OPERATIONS.md。

本项目额外用 SQLite 存储快照、节点、决策和回执，并提供 cleaning export/restore；这是可选的工程封装。其他项目可以直接读写普通 DoclingDocument JSON 与配套记录，不必复刻数据库。

正式切片、混合检索与嵌入是另一个任务，设计与操作入口分别在项目根目录 DESIGN.md 和 OPERATIONS.md。它消费已验收的文档及显式祖先关系，不是本 Skill 的最后一步。诊断切片可作为后续检查，但不作为 PDF 清洗完成的前置条件。现有 Skill 文本是一套执行指导，不应宣称新文档的任意格式接入、自动 Agent 调度和每项验收都已被代码完整实现。
