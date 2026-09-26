"""Local inspection gallery: escaped text, actual saved chunks, and unmodified PDF renders."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections import defaultdict

import pymupdf

from ..common import digest, read_json, write_json

SEED = "financial-data-quality-20260912-v1"


def pick_samples(details, pages, tables):
    selected = {}
    by_doc = defaultdict(list)
    for row in pages:
        by_doc[row["doc_id"]].append(row)

    def add(doc_id, page, reason):
        selected.setdefault((doc_id, page), []).append(reason)

    for entry in details.values():
        doc = entry["doc"]
        rows = by_doc[doc["doc_id"]]
        prose = [
            r
            for r in rows
            if r["text_nodes"] >= 5
            and r["native_characters"] > 1200
            and not r["tables"]
            and not r["pictures"]
            and r["page_number"] > 3
        ]
        if prose:
            chosen = min(prose, key=lambda r: digest([SEED, "prose", doc["doc_id"], r["page_number"]]))
            add(doc["doc_id"], chosen["page_number"], "固定种子的正文分层抽样")
        candidates = [
            r for r in tables if r["doc_id"] == doc["doc_id"] and r["cells"] >= 30 and r["header_cells"] > 0
        ]
        if candidates:
            chosen = min(candidates, key=lambda r: digest([SEED, "table", r["node_id"]]))
            add(doc["doc_id"], chosen["pages"][0], "固定种子的表格分层抽样")
        substantial = [
            r for r in rows if r["native_lexical_tokens"] >= 100 and r["native_token_recovery"] is not None
        ]
        if substantial:
            chosen = min(substantial, key=lambda r: (r["native_token_recovery"], r["page_number"]))
            add(doc["doc_id"], chosen["page_number"], "本文件文本重合率最低的非空页，作为风险样本")
        for row in rows:
            if row["native_token_recovery"] is not None and row["native_token_recovery"] < 0.9:
                add(doc["doc_id"], row["page_number"], "文本层词项重合低于 90% 的诊断页，不等于解析失败")
    known = {
        "jpmorgan_chase_2024": {
            11: "并排表格及标题路径",
            15: "跨页正文前半段",
            16: "跨页正文后半段",
            42: "金额及脚注对照",
            73: "空表对应原始组织架构图",
            96: "第二个空表图像回退",
        },
        "morgan_stanley_2024": {101: "已知货币符号与列对齐问题"},
        "bank_of_america_2024": {60: "字符范围异常及双栏正文拼接", 191: "Series B 完整条款的解析与切片定位"},
        "citigroup_2024": {
            152: "跨页字符范围异常的前页",
            153: "跨页字符范围异常的后页",
            157: "第二组跨页字符范围异常",
            158: "第二组跨页字符范围异常的后页",
            679: "长段拆片的来源精度",
            680: "长段拆片的来源精度",
        },
        "goldman_sachs_2024": {504: "短尾片的来源精度", 537: "长段拆片的来源精度", 557: "长段拆片的来源精度"},
    }
    for doc_id, pp in known.items():
        if doc_id in by_doc:
            for p, why in pp.items():
                if any(row["page_number"] == p for row in by_doc[doc_id]):
                    add(doc_id, p, why)
    return [{"doc_id": d, "page_number": p, "reasons": why} for (d, p), why in sorted(selected.items())]


def build_gallery(config, store, output, details, pages, chunks, tables):
    selected = pick_samples(details, pages, tables)
    images = output / "images"
    images.mkdir(exist_ok=True)
    by_doc = {v["doc"]["doc_id"]: v for v in details.values()}
    by_page = {(r["doc_id"], r["page_number"]): r for r in pages}
    by_chunk = {c["chunk_id"]: c for c in chunks}
    samples = []
    renderer = shutil.which("pdftoppm")
    for item in selected:
        entry = by_doc[item["doc_id"]]
        doc = entry["doc"]
        page_index = item["page_number"] - 1
        stem = f"{item['doc_id']}-{item['page_number']}"
        png = images / (stem + ".png")
        if not png.exists():
            if renderer:
                subprocess.run(
                    [
                        renderer,
                        "-f",
                        str(page_index + 1),
                        "-l",
                        str(page_index + 1),
                        "-r",
                        "120",
                        "-singlefile",
                        "-png",
                        doc["pdf_path"],
                        str(images / stem),
                    ],
                    capture_output=True,
                    check=True,
                    timeout=90,
                )
            else:
                with pymupdf.open(doc["pdf_path"]) as pdf:
                    pdf[page_index].get_pixmap(matrix=pymupdf.Matrix(120 / 72, 120 / 72), alpha=False).save(
                        png
                    )
        with pymupdf.open(doc["pdf_path"]) as pdf:
            rect = pdf[page_index].rect
            native = pdf[page_index].get_text("text", sort=True)
        nodes = [
            n
            for n in entry["nodes"]["structured"]
            if any(s["page_index"] == page_index for s in n["sources"])
        ]
        page_chunks = {}
        for kind in ("flat", "structured"):
            hits = [
                c for c in entry["chunks"][kind] if any(s["page_index"] == page_index for s in c["sources"])
            ]
            page_chunks[kind] = [
                {**c, "audit": by_chunk[c["id"]]}
                for c in sorted(
                    hits,
                    key=lambda c: (
                        min((s["bbox"] or [0, 0])[1] for s in c["sources"] if s["page_index"] == page_index),
                        c["id"],
                    ),
                )
            ]
        samples.append(
            {
                **item,
                "page_index": page_index,
                "doc_version": doc["id"],
                "pdf_sha256": doc["sha256"],
                "page_width": rect.width,
                "page_height": rect.height,
                "image": "images/" + png.name,
                "pdf": os.path.relpath(doc["pdf_path"], output).replace("\\", "/"),
                "native_text": native,
                "nodes": nodes,
                "chunks": page_chunks,
                "metrics": by_page[(item["doc_id"], item["page_number"])],
                "visual_review": "尚未记录逐页视觉复核结论",
            }
        )
        print(f"Rendered audit sample: {item['doc_id']} page {item['page_number']}", flush=True)
    data = {
        "index_ids": {
            kind: next(c["index_id"] for entry in details.values() for c in entry["chunks"][kind])
            for kind in ("flat", "structured")
        },
        "seed": SEED,
        "sampling": "6 documents: seeded prose, seeded table, lowest-overlap risk page, plus fixed known cases. Not a random accuracy estimate.",
        "renderer": "pdftoppm 120 dpi" if renderer else "PyMuPDF 120 dpi",
        "samples": samples,
    }
    write_gallery(output, data)
    return samples


def write_gallery(output, data):
    """Review notes may be reused only for the exact same saved indices."""
    path = output / "reviews.json"
    reviews = read_json(path) if path.exists() else {}
    if reviews.get("index_ids") == data.get("index_ids"):
        notes = {(r["doc_id"], r["page_number"]): r for r in reviews.get("pages", [])}
        for sample in data["samples"]:
            note = notes.get((sample["doc_id"], sample["page_number"]))
            if note:
                sample["visual_review"] = note["note"]
    write_json(output / "samples.json", data)
    payload = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    (output / "index.html").write_text(HTML.replace("__SAMPLE_DATA__", payload), encoding="utf-8")


HTML = r"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>金融 RAG · 数据质量对照</title>
<style>
:root{font:15px/1.65 system-ui,"Microsoft YaHei",sans-serif;color:#253348;background:#f4f6f9}*{box-sizing:border-box}body{margin:0}header{background:#142a3d;color:#f5f8fb;padding:20px 28px}h1{font-size:24px;margin:0 0 5px}header p{margin:0;color:#c0d1df}.tools{padding:16px 24px;position:sticky;top:0;background:#fff;z-index:3;border-bottom:1px solid #d8e0e8;display:flex;gap:12px;align-items:center;flex-wrap:wrap}select,button{font:inherit;padding:8px 12px;border:1px solid #afbecd;border-radius:6px;background:#fff;color:#24374a}select{min-width:260px;max-width:90vw}button{cursor:pointer}button.active{background:#174e6c;color:#fff}main{display:grid;grid-template-columns:minmax(400px,1.03fr) minmax(420px,1fr);gap:18px;padding:20px;align-items:start}.panel{background:white;border:1px solid #dce3eb;border-radius:9px;padding:18px}.meta{font-size:13px;color:#5b6c80;word-break:break-all}.page{position:relative;border:1px solid #ccd7e1;margin-top:12px}.page img{width:100%;display:block}.page svg{position:absolute;inset:0;width:100%;height:100%}.page rect{cursor:pointer;stroke-width:1;fill-opacity:.05;stroke-opacity:.55}.page rect:hover{fill-opacity:.25;stroke-opacity:1;stroke-width:2}.page rect.selected{fill-opacity:.28;stroke-width:2;stroke-opacity:1}.legend{display:flex;gap:16px;font-size:12px;margin:10px 0}.node,.chunk{border:1px solid #d3dfe8;border-radius:7px;padding:12px;margin:12px 0;background:#fcfdfe}.chunk h3,.node h3{font-size:14px;margin:0}.chunk pre,.node pre,pre#native{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.75 Consolas,"Microsoft YaHei",monospace;color:#263343;margin-bottom:0}.tag{display:inline-block;background:#e5edf5;color:#345777;border-radius:4px;padding:0 7px;font-size:12px;margin:4px 4px 0 0}.warn{background:#fff0d3;color:#7b530f}.muted{color:#718196}.tabs{display:flex;flex-wrap:wrap;gap:6px;margin:14px 0}.empty{padding:20px;background:#fff4df;border-radius:6px}h2{font-size:18px;margin:0 0 8px}a{color:#146286}.legend b{font-weight:500}.note{font-size:13px;background:#eef4f8;padding:12px;border-radius:6px}.selected-info{margin:10px 0}.count{font-weight:600}details summary{cursor:pointer;font-weight:600}.hidden{display:none}@media(max-width:1000px){main{grid-template-columns:1fr}.tools{position:static}}
</style><header><h1>金融 RAG：原页、解析与切片对照</h1><p>只读审查 · 不调用模型 · 保留真实原文和实际索引输入</p></header>
<div class="tools"><label>样本 <select id="sample"></select></label><button id="prev">上一页样本</button><button id="next">下一页样本</button><label><input type="checkbox" id="boxes" checked> 节点坐标</label><button id="clear">清除节点筛选</button></div>
<main><section class="panel"><h2 id="name"></h2><div class="meta" id="meta"></div><p class="note" id="why"></p><div class="legend"><b style="color:#257a51">绿：表格</b><b style="color:#8949b5">紫：图片</b><b style="color:#aa621f">橙：标题</b><b style="color:#287cac">蓝：正文等</b></div><a id="open-image" target="_blank">打开原页大图</a> · <a id="open-pdf" target="_blank">打开原 PDF</a><div class="page"><img id="page" alt="PDF 原始页面"><svg id="overlay"></svg></div><p class="meta" id="hash"></p></section>
<section class="panel"><h2>保存的数据实际是什么样</h2><p class="note">点击左侧区域可筛选对应节点和 B 切片。A 只保存页级来源。标题是原文上下文；表格数据用固定宽度原始 Markdown 展示，避免网页二次排版掩盖错列。</p><div class="meta" id="stats"></div><div class="selected-info" id="selected"></div><div class="tabs"><button data-tab="structured" class="active">B 结构切片</button><button data-tab="flat">A 普通切片</button><button data-tab="nodes">Docling 节点</button><button data-tab="native">PDF 文本层</button></div><div id="content"></div></section></main>
<script id="data" type="application/json">__SAMPLE_DATA__</script><script>
const dataset=JSON.parse(document.getElementById('data').textContent), samples=dataset.samples;
const $=id=>document.getElementById(id);let ix=0,tab='structured',nodeId=null;
function el(tag,text,cls){const v=document.createElement(tag);if(text!==undefined)v.textContent=text;if(cls)v.className=cls;return v}
samples.forEach((s,i)=>{const o=el('option',s.doc_id+' · 物理第 '+s.page_number+' 页');o.value=i;$('sample').append(o)});
function badge(text,warn=false){return el('span',text,'tag'+(warn?' warn':''))}
function draw(){const s=samples[ix];$('sample').value=ix;$('name').textContent=s.doc_id;$('meta').textContent=`物理页 ${s.page_number} · page_index ${s.page_index} · PDF 标签 ${s.metrics.pdf_page_label||'未设置'} · ${s.page_width} × ${s.page_height} points`;$('why').textContent=s.reasons.join('；')+'。'+s.visual_review;$('page').src=s.image;$('open-image').href=s.image;$('open-pdf').href=s.pdf+'#page='+s.page_number;$('hash').textContent='原 PDF SHA-256：'+s.pdf_sha256;const r=s.metrics.native_token_recovery;$('stats').textContent=`本页 Docling 节点 ${s.nodes.length}；A 切片 ${s.chunks.flat.length}；B 切片 ${s.chunks.structured.length}；文本层词项重合 ${(r===null?'N/A':(r*100).toFixed(1)+'%')}（诊断量，不是准确率）`;
const svg=$('overlay');svg.replaceChildren();svg.setAttribute('viewBox',`0 0 ${s.page_width} ${s.page_height}`);svg.style.display=$('boxes').checked?'block':'none';s.nodes.forEach(n=>n.sources.filter(x=>x.page_index===s.page_index&&x.bbox).forEach(src=>{const [x,y,r,b]=src.bbox;const box=document.createElementNS('http://www.w3.org/2000/svg','rect');box.setAttribute('x',x);box.setAttribute('y',y);box.setAttribute('width',r-x);box.setAttribute('height',b-y);const col=n.kind==='table'?'#257a51':n.kind==='picture'?'#8949b5':['title','section_header'].includes(n.kind)?'#aa621f':'#287cac';box.setAttribute('fill',col);box.setAttribute('stroke',col);if(n.id===nodeId)box.classList.add('selected');const title=document.createElementNS('http://www.w3.org/2000/svg','title');title.textContent=n.kind+' '+n.ref+'\n'+n.text.slice(0,100);box.append(title);box.addEventListener('click',()=>{nodeId=n.id;draw()});svg.append(box)}));show()}
function show(){const s=samples[ix],host=$('content');host.replaceChildren();$('selected').textContent=nodeId?'筛选节点：'+s.nodes.find(n=>n.id===nodeId)?.ref+' · '+nodeId.slice(0,12):'当前展示本页所有已保存入口';document.querySelectorAll('[data-tab]').forEach(b=>b.classList.toggle('active',b.dataset.tab===tab));if(tab==='native'){host.append(el('pre',s.native_text));host.firstChild.id='native';return}if(tab==='nodes'){const ns=s.nodes.filter(n=>!nodeId||n.id===nodeId);ns.forEach(n=>{const card=el('article',undefined,'node');card.append(el('h3',n.kind+' · '+n.ref));card.append(el('div','标题路径：'+n.headings.join(' → '),'meta'));card.append(el('div',`父节点 ${n.parent_ref} · 阅读顺序 ${n.reading_order??'结构容器/图片内部'} · 深度 ${n.depth}`,'meta'));if(!n.text.trim())card.append(el('p','没有解析文字；原始图像仍可回读。','empty'));else card.append(el('pre',n.text));const d=el('details');d.append(el('summary','查看完整原始节点及来源'));d.append(el('pre',JSON.stringify({sources:n.sources,raw:n.raw},null,2)));card.append(d);host.append(card)});return}
const cs=s.chunks[tab].filter(c=>tab==='flat'||!nodeId||c.node_ids.includes(nodeId));host.append(el('p',`${cs.length} 条实际切片`,'count'));if(!cs.length)host.append(el('p','该节点没有独立 B 检索入口。可以切换到 Docling 节点或原页检查；图片描述属于后续 C 的处理。','empty'));cs.forEach((c,i)=>{const card=el('article',undefined,'chunk');card.append(el('h3',`${i+1}. ${c.kind} · ${c.id.slice(0,14)}`));card.append(badge(c.audit.tokens+' token',c.audit.overflow));card.append(badge('来源物理页 '+c.audit.pages.join(', ')));card.append(badge(c.node_ids.length+' 个节点'));if(c.audit.contains_split_multipage_node)card.append(badge('含被切分的跨页节点：页来源可能偏宽',true));card.append(el('div',c.headings.length?'标题路径：'+c.headings.join(' → '):'无结构标题上下文','meta'));card.append(el('pre',c.text));const d=el('details');d.append(el('summary','查看切片 ID、节点与坐标'));d.append(el('pre',JSON.stringify({id:c.id,node_ids:c.node_ids,sources:c.sources},null,2)));card.append(d);host.append(card)})}
$('sample').onchange=e=>{ix=Number(e.target.value);nodeId=null;draw()};$('prev').onclick=()=>{ix=(ix+samples.length-1)%samples.length;nodeId=null;draw()};$('next').onclick=()=>{ix=(ix+1)%samples.length;nodeId=null;draw()};$('boxes').onchange=draw;$('clear').onclick=()=>{nodeId=null;draw()};document.querySelectorAll('[data-tab]').forEach(b=>b.onclick=()=>{tab=b.dataset.tab;show()});if(samples.length)draw();
</script></html>"""
