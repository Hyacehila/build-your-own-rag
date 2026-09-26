# 当前 v2 操作手册

[项目首页](README.md) · [设计](DESIGN.md) · [验收](VALIDATION.md)

以下命令在本项目目录执行。先安装 Python 3.11、`uv`，运行 `uv sync --locked`；版本以 `.python-version`、`pyproject.toml` 和 `uv.lock` 为准。工作目录由[config.toml](config.toml)定义；需要隔离恢复时复制它，仅修改 `work_dir`。`.env.example` 列出可选密钥名，`.env` 被 Git 忽略。原始六份 PDF 不在 Git 中，恢复时 `--pdf-dir` 应直接指向经快照校验的 PDF 目录。发布 ID、校验值和阶段结果统一查[验收记录](VALIDATION.md)。异地全量实验的交付清单、前置条件和 AI 启动提示词见[实验交接盘点](EXPERIMENT_HANDOFF.md)。

## 1. 无 API：查看、恢复与核查

已有工作库时，运行 `uv run python -m financial_rag.retrieval_release check` 检查当前 A/B；它用已缓存向量做检索冒烟，不产生新模型请求。需要从交接物恢复到新工作目录时：

```powershell
Copy-Item config.toml config.restore.toml
# 在 config.restore.toml 中把 work_dir 改为全新的目录。
uv run python -m financial_rag.cleaning restore --config config.restore.toml --database handoff/finance-facts-v2.sqlite --output .local/facts-v2-restored --pdf-dir D:/verified-pdfs
uv run python -m financial_rag.cleaning check --config config.restore.toml --output .local/facts-v2-restored
uv run python -m financial_rag.retrieval_release prepare --config config.restore.toml --database handoff/finance-retrieval-v2.sqlite --pdf-dir D:/verified-pdfs
uv run python -m financial_rag.retrieval_release check --config config.restore.toml
```

事实与检索快照用途不同，应分别验证；`prepare` 会核对来源和已有切片/向量，不重新调用神经解析或嵌入。`cleaning restore` 的 `--output` 与检索 `work_dir` 也应分开。已发布交接文件仅为两个 SQLite、两个检索验收 JSON 和 LFS 规则。

若要在 Docling Studio 看整理后的文档，先拥有可回读的事实目录及 PDF，然后执行：

```powershell
powershell -ExecutionPolicy Bypass -File tools/docling-studio/Start-DoclingStudio.ps1 -PrepareOnly -FactsRoot .local/facts-v2-release -WorkspaceRoot .local/docling-studio/workspace-v2
powershell -ExecutionPolicy Bypass -File tools/docling-studio/Start-DoclingStudio.ps1 -FactsRoot .local/facts-v2-release -WorkspaceRoot .local/docling-studio/workspace-v2
```

第二条启动本地界面，脚本输出地址。导入的是查看副本，不会改变事实发布。首次准备可能下载锁定的 Studio 源码与依赖；这不是付费模型调用。若查看新恢复的事实，把 `-FactsRoot` 指向对应恢复目录。

## 2. 当前 A/B：从 v2 事实构建

已验收事实库和原 PDF 就绪后，可在独立 `work_dir` 从头建立 A/B。嵌入命令可能调用付费服务；先用 `estimate-embedding` 核对规模。

```powershell
uv run python -m financial_rag.retrieval_release prepare --database handoff/finance-facts-v2.sqlite --pdf-dir D:/verified-pdfs
uv run finance-rag parse --parser flat
uv run finance-rag chunk --index flat
uv run finance-rag chunk --index structured
uv run python -m financial_rag.retrieval_release check --without-vectors
uv run finance-rag estimate-embedding
uv run finance-rag embed --index flat
uv run finance-rag embed --index structured
uv run python -m financial_rag.retrieval_release check
```

仅在检查通过、目标路径不存在时才导出：`uv run python -m financial_rag.retrieval_release export --database handoff/finance-retrieval-v2.sqlite`。事实候选完成审阅后用 `uv run python -m financial_rag.cleaning check` 与 `export`；此发布路径的具体审阅要求在[Skill 项目适配](.agents/skills/clean-financial-documents/references/project-adapter.md)。如需从原始工作库运行 `cleaning prepare`，必须显式给出 `--source-work-dir`，不能把当前检索工作库误当原始输入。

## 3. 有限 API 小样

先复制 `.env.example` 为 `.env` 并填写自己的 `DASHSCOPE_API_KEY`、`DEEPSEEK_API_KEY`。运行 `uv run finance-rag doctor` 做本地契约检查；`doctor --live` 会调用外部服务，须先在专用小样配置中设置 `validation.ledger`，默认共享配置不满足这一条件。`uv run --locked python tools/live_preflight.py` 对虚构夹具各跑 A/B/C/D 一题，对当前真实 v2 A/B 各跑一题；脚本的 DeepSeek 请求上限为 20，并在独立账本中记录尝试。该脚本固定读取项目 `config.toml`，不接受 `--config`；它不是全量启动器。再次运行相同 run ID 只适合续跑相同输入；新目的使用新的 `--run-id` 和 `--batch`。它不做答案质量评分。

## 4. 后续正式实验

先为真实六份财报完成 C 的图表增强、切片与向量，再让 D 复用 C。绑定经验证的英文问题、页级标注和版本后，才能安排 309 题四组运行、评分和报告。`finance-rag enrich`、`chunk --index enriched`、`embed --index enriched`、`run`、`evaluate`、`report` 是对应代码入口；这些步骤尚未构成已完成的 v2 发布流程，也会产生外部 API 用量。
