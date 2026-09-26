# 历史清洗配置

`cleaning-v1/` 保留旧版六份财报的来源绑定和清洗决定，仅供历史对照与复现。当前清洗入口默认使用 `profiles/cleaning-v2/`，不会自动读取此目录。

旧命令若显式指定了 `--profiles profiles/cleaning-v1`，复现时改为 `--profiles profiles/archive/cleaning-v1`，并使用匹配的 PDF 和原始快照。归档只改变文件位置，未修改 JSON 内容或 v2 发布数据库。
