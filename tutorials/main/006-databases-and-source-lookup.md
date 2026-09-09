# 006 · 数据库存储与结构化回查

简体中文 | 英文译文待补充

上一章：[从召回到回答：查询流程与 Agentic RAG](005-retrieval-and-agentic-rag.md) · [返回总目录](../../README.md#主线教程目录) · 下一章：尚未发布

第五篇已经介绍了从检索到回答的查询流程。但在真正实现时，资料应该如何存储，如何找到对应版本的原文，又如何按需扩大读取范围，找到它的上下文？

这些问题会影响数据库的组织方式。基于我们的独特需求，构建一个足够高效的数据库系统，就是本篇想聊的问题。

我们先聊一聊我比较常用的 SQLite 和 PostgreSQL，再讨论如何在这两个关系型数据库中设计所需的数据表。这也能解释，为什么我会选择在关系型数据库中加入全文检索与向量扩展，来组织这套 RAG 系统。

*本文不展开 NoSQL 数据库的选型；关系型数据库也能通过 JSON 等能力保存结构灵活的数据。*

## 数据库：从 SQLite 和 PostgreSQL 说起

关系型数据库用表保存记录，再通过标识把记录关联起来。放到这个 RAG 系统里，需要组织的包括文档及版本、原文节点、增强后的检索资料，以及检索资料与原文之间的对应关系。这些关联决定了搜索命中以后，我们能否读到正确的来源。

### SQLite：先把一个应用里的资料管好

SQLite 嵌入在应用中运行，不需要另起一个数据库服务。对于本地工具、个人知识库，或者写入并发不高的单机应用，我会先考虑它，部署和管理都比较简单。不过，同一个数据库文件同一时刻只能有一个写入者，是否合适还要看应用的写入方式。我会重点看有多少任务需要同时写入、每次写入持续多久，以及它们能否排队，而不只看有多少用户同时访问。

### PostgreSQL：把数据库作为独立服务

PostgreSQL 通常作为独立的数据库服务运行，应用通过连接访问它。需要多个应用实例共享数据、处理更多并发读写，或者集中管理访问权限时，我会进一步考虑 PostgreSQL。它提供事务、索引和访问控制等能力，也需要相应的部署与维护。

对我来说，选择哪一个主要取决于应用怎样部署、资料怎样更新，以及愿意承担多少维护工作。

## SQLite 和 PostgreSQL 提供什么能力

我们来聊一聊这两个数据库主要提供的能力范围以及它们的边界。这样方便对这两个数据库不熟悉的读者判断自己要不要用它们。

### Foundation

SQLite 可以通过应用程序里的库直接使用。比如下面的代码：

```python
import sqlite3

conn = sqlite3.connect("database.db")
```

这段代码直接打开本地数据库文件，不需要经过网络服务，也不需要维护独立的数据库服务，部署与管理成本较低，适合端侧 App 等场景。

SQLite 虽然用起来简单，但依旧提供常用的 SQL 查询、关联和事务能力，也支持通过 B-tree 索引减少查询开销。它支持 ACID 事务，核心语法也区别不大。

判断数据库的并发需求时，我会把读取和写入分开看。很多用户同时查询知识库，都不涉及什么修改；本地 App 后台导入、更新和日志任务也可能产生并发写入。SQLite 在 WAL 模式下允许读写并行，但同一数据库仍只有一个写入者。PostgreSQL 通过多版本并发控制（Multi-Version Concurrency Control，MVCC）让普通读取使用数据快照，减少读写之间的阻塞，只有多个事务修改同一行时，仍可能发生等待。

PostgreSQL 提供多种索引，用来加速不同类型的查询。B-tree 常用于等值、范围和排序；GIN 是倒排索引，适合全文检索，以及部分 JSONB、数组查询；BRIN 保存连续数据块的摘要，适合字段值与数据物理排列顺序有较强关联的场景，按时间持续追加的数据是一个常见例子。跨表检索一个 Chunk 一般会利用多种索引。

对 RAG 项目来说，关系型数据库还有一个很实用的能力：通过事务和约束维护资料之间的关联。事务可以让一组数据库修改一起提交，或者在失败时一起撤销；主键和唯一约束避免重复标识，外键则可以检查被引用的记录是否存在。这样，在更新文档、节点和检索资料时，就能尽可能保证一致性。

### Vector Index

虽然我不太喜欢嵌入模型以及对应的向量检索技术，但向量检索仍是 RAG 不可缺少的一部分。

在 PostgreSQL 中，可以通过 pgvector 扩展增加向量存储与检索能力。在服务器已安装 pgvector 的前提下，用下面的 SQL 在目标数据库中启用扩展。

```sql
CREATE EXTENSION vector;
```

然后就能在表中定义向量列。下面只是能力示意，尚未加入后文回查所需的版本、节点关系和完整约束：

```sql
CREATE TABLE chunks (
    id BIGSERIAL PRIMARY KEY,
    tenant_id BIGINT,
    document_id BIGINT,
    content TEXT,
    embedding VECTOR(1024),
    metadata JSONB
);
```

这张表把向量、正文和元数据放在同一条记录中。pgvector 还提供多种距离计算方式，供查询时使用。

pgvector 默认进行精确最近邻搜索，也可以建立 HNSW 或 IVFFlat 索引，用部分召回率换取查询速度。向量类型的存储上限和近似最近邻（ANN）索引的维数上限需要分开看：当前 `vector` 类型最多保存 16000 维向量，但它的 HNSW / IVFFlat 索引最多支持 2000 维；`halfvec` 对应的这两类索引则支持到 4000 维。因此，选择嵌入模型时，还要一起确认向量类型和索引方案。

使用向量扩展，可以在同一条 SQL 中表达过滤条件与向量距离排序。

```sql
SELECT *
FROM chunks
WHERE tenant_id = 123
  AND document_id IN (...)
ORDER BY embedding <=> :query_embedding
LIMIT 20;
```

过滤条件可以与向量查询写在同一条 SQL 中，但实际执行顺序由查询计划决定。使用 pgvector 的近似索引时，可能先扫描向量候选，再应用过滤条件，因此即使符合条件的数据足够，也可能返回不足 `LIMIT` 指定数量的结果。

SQLite 可以通过 sqlite-vec 扩展提供向量检索能力，也提供了基础的检索能力。sqlite-vec 支持在 `vec0` 的 KNN 查询中加入受支持的 metadata 条件，也可以通过 partition key 缩小搜索范围。

### JSONB

文档型数据库常用类似 JSON 的结构组织记录。关系型数据库也可以在表的一列中保存 JSON，用来容纳字段不完全固定、带有嵌套关系的数据。

SQLite 和 PostgreSQL 都提供 JSONB，但两者的二进制格式并不兼容。SQLite 的 JSONB 主要减少重复解析 JSON 的开销，通常也更节省空间；它的大多数 JSON 操作仍是 O(N)。对经常使用的 JSON 字段，还可以考虑表达式索引，或者提取为普通列。

PostgreSQL 的 JSONB 可以结合 GIN，加速键存在、包含关系和部分 JSONPath 查询。JSONB 适合保存形态多变的附加信息；经常用于关联、过滤和排序的文档标识、版本及节点关系一般应该考虑放回关系型数据库中。

### Full Text Search

PostgreSQL 提供全文检索（Full Text Search，FTS），SQLite 也可以通过 FTS5 使用全文检索，其背后基于倒排索引。将全文检索和稠密向量检索的结果通过 RRF 融合，再结合元数据过滤，是我通常优先尝试的组合，

有全文检索能力，不意味着默认配置就能处理好中文。全文检索首先要把文档和查询拆成可匹配的词项。SQLite FTS5 默认的 `unicode61` 根据字符类别划分词项，并不会自动完成中文词语切分；PostgreSQL 也需要考虑解析器和词典配置。对于中文、专业术语以及中英文混排的资料，要检查实际生成了哪些词项，再决定是否增加分词处理或相关扩展。

资料入库与用户查询还需要采用相互兼容的处理方式。如果资料中的术语被拆开，而查询保留了完整名称，即使原文里明明有答案，也可能无法通过关键词召回。往往需要在初步 Demo 搭建后进行一些 trace 进行检查。

## 如何在数据库中构建高效检索层

搜索可以让我们找到可用的 Chunk，但它们只是检索入口。接下来需要找到原文节点，再决定是否读取邻近段落、所属章节或关联图注。我希望数据库能直接支持这些动作，避免每次只取几个节点，却要加载和反序列化整份文档。

当节点回查成为高频操作时，那就应该考虑在导入阶段多做一次整理，把需要查询的关系存成数据库可以直接访问的记录。

### 完整文档、原文节点和检索资料分别保存

完整的 DoclingDocument 仍然保留，作为这一解析版本的完整记录。数据库中的节点表是从它生成的查询用副本，后续修改整理规则时可以重建。原始文件也应保留，如果后面发现解析质量有问题，就可以从头再来。

我用四张核心表组织这件事：

| 表 | 保存什么 | 查询时负责什么 |
| --- | --- | --- |
| `documents` | 文档版本、原始来源和完整文档对象 | 确认来源版本，必要时加载完整对象 |
| `doc_items` | 原文节点及其阅读顺序、父子关系、完整节点数据 | 按节点读取，并扩展上下文 |
| `chunks` | 内容、摘要与最终检索文字 | 提供全文和向量检索的资料 |
| `chunk_items` | Chunk 与原文节点之间的多对多关系 | 把搜索命中转换成原文定位 |

一个 Chunk 可能合并几个原文节点；一个长节点也可能被拆成多个 Chunk。所以这里单独放一张 `chunk_items`，而不是在节点上只存一个 `chunk_id`。全文检索和向量存储继续关联到 Chunk，索引方式同前。

节点的完整定位由 `doc_id + version_id + item_ref` 组成。`item_ref` 对应 Docling 的 `self_ref`，只在这份文档对象中有意义。这里的 `version_id` 绑定一次确定的解析与整理结果，保证版本正确跟踪。

### 四张表的基本形状

下面以 SQLite 为例说明关系。

```sql
PRAGMA foreign_keys = ON;

CREATE TABLE documents (
    doc_id          TEXT NOT NULL,
    version_id      TEXT NOT NULL,
    source_uri      TEXT,
    source_hash     TEXT,
    docling_jsonb   BLOB NOT NULL,
    PRIMARY KEY (doc_id, version_id)
);

CREATE TABLE doc_items (
    doc_id          TEXT NOT NULL,
    version_id      TEXT NOT NULL,
    item_ref        TEXT NOT NULL,
    item_type       TEXT NOT NULL,
    label           TEXT,
    text            TEXT,
    reading_order   INTEGER CHECK (reading_order >= 0),
    parent_ref      TEXT,
    sibling_order   INTEGER CHECK (sibling_order >= 0),
    depth           INTEGER NOT NULL CHECK (depth >= 0),
    section_ref     TEXT,
    section_path    TEXT,
    raw_jsonb       BLOB NOT NULL,
    PRIMARY KEY (doc_id, version_id, item_ref),
    FOREIGN KEY (doc_id, version_id)
        REFERENCES documents (doc_id, version_id),
    FOREIGN KEY (doc_id, version_id, parent_ref)
        REFERENCES doc_items (doc_id, version_id, item_ref)
        DEFERRABLE INITIALLY DEFERRED,
    FOREIGN KEY (doc_id, version_id, section_ref)
        REFERENCES doc_items (doc_id, version_id, item_ref)
        DEFERRABLE INITIALLY DEFERRED
);

CREATE TABLE chunks (
    doc_id          TEXT NOT NULL,
    version_id      TEXT NOT NULL,
    chunk_id        TEXT NOT NULL,
    content         TEXT NOT NULL,
    summary         TEXT,
    retrieval_text  TEXT NOT NULL,
    PRIMARY KEY (doc_id, version_id, chunk_id),
    FOREIGN KEY (doc_id, version_id)
        REFERENCES documents (doc_id, version_id)
);

CREATE TABLE chunk_items (
    doc_id          TEXT NOT NULL,
    version_id      TEXT NOT NULL,
    chunk_id        TEXT NOT NULL,
    item_ref        TEXT NOT NULL,
    item_order      INTEGER NOT NULL CHECK (item_order >= 0),
    PRIMARY KEY (doc_id, version_id, chunk_id, item_ref),
    UNIQUE (doc_id, version_id, chunk_id, item_order),
    FOREIGN KEY (doc_id, version_id, chunk_id)
        REFERENCES chunks (doc_id, version_id, chunk_id),
    FOREIGN KEY (doc_id, version_id, item_ref)
        REFERENCES doc_items (doc_id, version_id, item_ref)
);
```

`doc_items` 这个表名是工程上的简写。除了文本、图片、表格等内容项，我也会存入 `body`、`furniture` 和 group 等结构节点，保证父节点可以被找到；不直接进入正文窗口的节点，其 `reading_order` 可以为空。`parent_ref` 表示直接父节点，`sibling_order` 表示它在父节点 `children` 中的位置。延迟外键检查允许我们在同一个事务中写入互相引用的节点，提交时再检查引用是否存在。

`section_ref` 是为了查询方便预先计算的所属章节引用，本例取最近的章节标题，标题自身归入自己；`section_path` 用 JSON 文本保存标题路径，主要用于展示。找不到明确章节关系则留空。

`raw_jsonb` 保存该节点完整的序列化数据，包括原有引用、`prov` 和表格的 `data` 等。页码、坐标和字符范围可能有多条来源记录，也可能不存在，以后如果经常按页查询，再为这些定位信息单独建表。

这样，常用的标识与关系是普通列，复杂内容仍用 JSONB 保存。表格、图片以及增强信息都像普通的元素一样被保存。

### 导入时，把顺序和树关系保存下来

SQL 查询结果没有可以依赖的天然行顺序。DoclingDocument 的正文阅读顺序则来自 `body` 树和各节点 `children` 的排列。导入时生成 `reading_order`，就是把这个顺序变成可查询的字段。

沿用前面的虚构文档“星港项目上线与值班安排”，在 `xinggang-launch-duty / teaching-v1` 中，可以得到下面的教学序列。`#/body` 作为结构根另存一行，不占正文序号。

| `reading_order` | `item_ref` | 内容 | `parent_ref` |
| --- | --- | --- | --- |
| 0 | `#/texts/0` | 星港项目上线与值班安排 | `#/body` |
| 1 | `#/texts/1` | 上线值班（章节标题） | `#/texts/0` |
| 2 | `#/texts/2` | 段落 A：上线窗口与交接时间 | `#/texts/1` |
| 3 | `#/texts/3` | 段落 B：负责人无法响应时联系备班 | `#/texts/1` |
| 4 | `#/tables/0` | 本周值班表 | `#/texts/1` |
| 5 | `#/pictures/0` | 告警升级流程图 | `#/texts/1` |
| 6 | `#/texts/4` | 图注：告警升级流程 | `#/texts/1` |
| 7 | `#/texts/5` | 回滚安排（章节标题） | `#/texts/0` |
| 8 | `#/texts/6` | 段落 C：回滚操作 | `#/texts/5` |

表格排在两个正文段落后面，图片又排在表格后面，跨类型的顺序就保留下来了。本例图注在正文树中是图片的兄弟节点，图片还通过 `captions` 引用它。这是图表和图注单独的一种关系。

一次导入可以分成下面几步：

1. 固定文档对象及其版本，保留来源文件与解析、整理配置。将完整对象和所有需要回查的节点写入对应版本。
2. 沿树读取 `parent`、`children`，记录父节点、兄弟顺序和深度。按约定的正文遍历范围，为内容节点生成从 0 开始的连续 `reading_order`；结构容器与页眉页脚不占本例的正文序号，图片内部识别项也不默认展开为正文。
3. 根据已有层级计算 `section_ref` 和标题路径。图注等关联目标即使不在正文序列中，也要能通过引用定位，不能因为没有序号就丢掉。
4. 生成 Chunk 后，把它对应的来源节点写入 `chunk_items`。使用 Docling Chunk 元数据时，从 `meta.doc_items` 中的节点对象提取 `self_ref`，并去除同一 Chunk 内的重复引用。
5. 检查引用、父子关系、顺序和来源数据是否一致，准备完成后再让这一版本对查询可见。模型调用放在提交过程之外，节点副本与检索资料都从同一版本生成。

### 从 Chunk 命中定位到原文节点

搜索返回 Chunk 后，先通过映射表回到原文。下面的 `:doc_id`、`:version_id` 和 `:chunk_id` 都是应用绑定的参数；例如第五篇的 `text-001` 绑定到 `xinggang-launch-duty / teaching-v1`。

```sql
SELECT ci.item_ref, ci.item_order,
       di.item_ref AS resolved_ref,
       di.item_type, di.text, di.reading_order,
       di.parent_ref, di.section_ref,
       json(di.raw_jsonb) AS raw_json
FROM chunk_items AS ci
LEFT JOIN doc_items AS di
  ON di.doc_id = ci.doc_id
 AND di.version_id = ci.version_id
 AND di.item_ref = ci.item_ref
WHERE ci.doc_id = :doc_id
  AND ci.version_id = :version_id
  AND ci.chunk_id = :chunk_id
ORDER BY ci.item_order;
```

这里按 `item_order` 返回该 Chunk 的来源排列。`text-001` 对应 `#/texts/2` 和 `#/texts/3`，`table-001` 对应 `#/tables/0`，于是得到实际原文。节点的结构数据已经在 `raw_jsonb` 中，也可以从数据库中查询全部原文。

### 上下文回查：按阅读顺序取一个窗口

为正文顺序、直接子节点和所属章节分别建立索引：

```sql
CREATE UNIQUE INDEX idx_doc_reading_order
ON doc_items (doc_id, version_id, reading_order);

CREATE UNIQUE INDEX idx_doc_children
ON doc_items (doc_id, version_id, parent_ref, sibling_order);

CREATE INDEX idx_doc_section_order
ON doc_items (doc_id, version_id, section_ref, reading_order);
```

前两项分别约束同一文档版本中的正文序号，以及同一父节点下的兄弟位置不重复。没有正文序号的结构节点可以保留 `NULL`。

假设命中段落 B，读取其节点记录后知道 `reading_order = 3`、`section_ref = '#/texts/1'`。两侧各扩展两个正文节点，就把窗口设为 1 到 5；同时要求留在当前章节：

```sql
SELECT item_ref, item_type, text, reading_order,
       json(raw_jsonb) AS raw_json
FROM doc_items
WHERE doc_id = :doc_id
  AND version_id = :version_id
  AND section_ref IS :section_ref
  AND reading_order BETWEEN :start_order AND :end_order
ORDER BY reading_order;
```

图注在序号 6，虽然不在窗口内，仍可以通过图片的 `captions` 引用补读。示例查询没有表达这一点。

换成以流程图为锚点，`start_order = 3`、`end_order = 7`，章节仍限定为 `#/texts/1`，结果就只有序号 3、4、5、6；序号 7 已经属于其他章节，不会进入本次窗口。这正是正文范围和章节边界一起参与查询的作用。

### 父子回查：直接孩子和整棵子树分开处理

从命中节点读出 `parent_ref` 后，用同一文档、同一版本下的 `item_ref` 等值查询就能取得父节点。再把这个引用绑定为 `:parent_ref`，可以读取它的直接孩子：

```sql
SELECT item_ref, item_type, text, sibling_order,
       json(raw_jsonb) AS raw_json
FROM doc_items
WHERE doc_id = :doc_id
  AND version_id = :version_id
  AND parent_ref = :parent_ref
ORDER BY sibling_order;
```

在示例中，以 `#/texts/1` 为父节点，会返回段落 A、段落 B、值班表、流程图和图注。命中节点本身也在其中，需要兄弟节点时再排除它。父节点如果是没有正文的 group，还需要继续选择它的孩子。

读取整棵子树则需要递归。可以由程序逐层批量读取直接孩子，也可以用 SQLite 和 PostgreSQL 都支持的递归 CTE：以目标节点为起点，反复连接下一层的 `parent_ref`。不论采用哪种方式，都要设置展开深度、节点数和证据长度上限；读取结束后，正文内容按 `reading_order` 排列，结构节点用于恢复层级。

下面给出 SQLite 的子树读取示意。`:root_ref` 是展开起点，`:max_depth` 是最多向下走几层，`:max_nodes` 是递归集合最多接收多少节点，包含起点和结构容器。

```sql
WITH RECURSIVE subtree(item_ref, hops) AS (
    SELECT item_ref, 0
    FROM doc_items
    WHERE doc_id = :doc_id
      AND version_id = :version_id
      AND item_ref = :root_ref

    UNION ALL

    SELECT child.item_ref, subtree.hops + 1
    FROM doc_items AS child
    JOIN subtree ON child.parent_ref = subtree.item_ref
    WHERE child.doc_id = :doc_id
      AND child.version_id = :version_id
      AND subtree.hops < :max_depth
    LIMIT :max_nodes
)
SELECT di.item_ref, di.parent_ref, di.reading_order,
       subtree.hops, json(di.raw_jsonb) AS raw_json
FROM subtree
JOIN doc_items AS di ON di.item_ref = subtree.item_ref
WHERE di.doc_id = :doc_id
  AND di.version_id = :version_id
  AND di.reading_order IS NOT NULL
ORDER BY di.reading_order;
```

本例从 `#/texts/1` 开始，允许向下一层、节点上限足够时，会返回序号 1 到 6：章节标题加它的五个孩子。与直接孩子查询相比，子树查询包含起点，并能在深度预算允许时继续穿过嵌套容器。

父子展开、正文窗口和图注回读可能重叠。读取层应统一按来源定位去重，保留表格整体，并明确标记因预算而截断的内容，然后交由回答或 Agentic 决策。

## 图表内容与完整对象放在哪里

表格的单元格数据和表格截图，需要分开考虑。单元格文字、行列位置、合并关系和来源定位，都可以随表格节点存进 `raw_jsonb`。普通问答需要的是这些数据；只有做视觉核对、展示原始布局等操作时，才需要加载截图。

图片节点及原有图注保存在事实层，模型生成的描述保存在增强层。PNG、JPEG、页面图像和原始 PDF 等二进制资源，可以考虑放在文件系统中进行引用，也可以补充 `assets` 使用数据库存储。若选择单个 SQLite 文件交付，也可以把资源存成 BLOB，只是要相应承担文件体积和备份、迁移的成本。

在这套方案里，普通回查主要访问节点行，需要表格结构时读取节点 JSON，需要图片时按资源引用加载，完整导出或重新整理时再读取完整文档。

在真实实践中，这套表方案比全部以来文件系统和 JSON 相关的索引要更快，且收益随着规模的提高而提高。单份文档越大、需要的节点占比越小，这种组织方式的优势往往越明显。

根据我的一些实践经验，对于 1MB 左右的 JSON，SQLite 基本回查可能只需要 0.1ms ，文件吸引解析则需要 2ms，随着文件规模继续扩大，文件读取呈现比线性更快的增长，而 SQLite 都在 1ms 以下，部分情况下耗时可能是几百倍的差距。

*耗时均以 P50 计*

## 最后几个容易混淆的问题

### 数据库里的表没有顺序，为什么还能找上下文？

无序的是查询结果没有默认的阅读顺序，并不妨碍把顺序存成数据。`reading_order` 记录已经从文档树得到的顺序，查询条件决定读取范围，`ORDER BY` 决定输出顺序。这与数据库索引机制无关，而是一个单独的顺序列。

### 能不能用 ID，或者 `self_ref` 末尾的数字代替顺序？

不能把这些东西默认当成正文顺序。ID 负责区分记录，`self_ref` 负责定位对象内的节点，`reading_order` 才负责描述正文位置。`#/texts/3` 后面可能是 `#/tables/0`，只对文本数组下标加一，就会漏掉表格和图片。

### 有了 JSONB，为什么还要拆出节点表？

JSONB 减少的是解析 JSON 文本的开销，节点表解决的是按节点、章节和阅读位置查询的问题。可以直接用 JSON 路径读取完整对象，但把高频关系放在列里，更便于建立对应索引。

### 保存完整文档和节点表，会不会变成两份互相冲突的事实？

它们有不同的读取用途，但应来自同一个确定版本。完整对象保留结构全貌，节点表是它的查询用副本；不能一边改节点正文，一边让完整对象继续保存另一份内容。需要修正时，更新权威来源或整理结果，生成新版本并重建相关副本。摘要和图片描述还要与原文区分，增强层与事实层依旧严格隔离。

### 图片和表格都放数据库里，会不会更方便？

表格结构放数据库很适合本章的节点回查，但表格截图和图片二进制不一定需要与正文放在同一个数据库文件里。要求单文件携带的离线知识库、小图或缩略图，可以考虑 BLOB；资源较多且需要独立分发时，可以使用外部资源存储。

---

本章原创正文采用 [CC BY 4.0](../../LICENSE-DOCS)。
