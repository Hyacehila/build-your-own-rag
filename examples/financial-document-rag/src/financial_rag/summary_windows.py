"""Question-independent five-original-block summary windows."""
import json
from collections import defaultdict

from .common import RagError, digest
from .facts import BODY_KINDS

CONTRACT = "same-section-five-blocks-overlap-one-v1"


def original_chunks(store, index_id):
    # put_index persists HybridChunker's original sequence. IDs are hashes, not positions.
    return [json.loads(r[0]) for r in store.db.execute(
        'SELECT payload FROM chunks WHERE index_id=? ORDER BY rowid', (index_id,))]


def window_slices(size):
    start = 0
    while start < size:
        end = min(size, start + 5)
        yield start, end
        if end == size:
            break
        start += 4


def ordered_sections(chunks, nodes):
    """Use real section identities and original spans, never hash-ID ordering."""
    groups = defaultdict(list)
    order_in_index = {c['id']: i for i, c in enumerate(chunks)}
    for chunk in chunks:
        if chunk['kind'] != 'text' or chunk.get('enrichment_id'):
            continue
        members = [nodes[n] for n in chunk['node_ids']]
        body = [n for n in members if n['kind'] in BODY_KINDS]
        if not body:
            continue
        identities = {(n['parse_id'], n.get('section_id')) for n in body}
        if len(identities) != 1:
            raise RagError('A body chunk crosses reviewed section identities')
        key = next(iter(identities))
        positions = [(n['reading_order'], n['id']) for n in body]
        spans = chunk.get('fragment_spans', [])
        offset = min((s['orig_start'] for s in spans), default=0)
        groups[key].append((min(positions)[0], offset, chunk))
    output = {}
    for key, entries in groups.items():
        # Same-node pieces may lack explicit spans; locate each exact body substring.
        for i, (order, offset, chunk) in enumerate(entries):
            if not chunk.get('fragment_spans'):
                header = '\n'.join(chunk.get('headings', []))
                text = chunk['text'][len(header):].lstrip('\n') if header and chunk['text'].startswith(header) else chunk['text']
                node_text = '\n'.join(nodes[n]['text'] for n in chunk['node_ids'])
                position = node_text.find(text)
                if position >= 0:
                    entries[i] = (order, position, chunk)
        entries.sort(key=lambda x: (x[0], x[1], order_in_index[x[2]['id']]))
        output[key] = [c for _, _, c in entries]
    return output


def body_windows(store, base):
    nodes = {n['id']: n for pid in base['parse_ids'] for n in store.nodes(pid)}
    output = []
    for (parse_id, section), chunks in ordered_sections(original_chunks(store, base['id']), nodes).items():
        # Root-level text must not jump over an intervening reviewed section.
        runs = [chunks]
        if section is None:
            runs = [[]]
            for c in chunks:
                if runs[-1]:
                    left = max(nodes[n]['reading_order'] for n in runs[-1][-1]['node_ids'])
                    right = min(nodes[n]['reading_order'] for n in c['node_ids'])
                    if any(n['parse_id'] == parse_id and n['kind'] == 'section_header'
                           and left < (n.get('reading_order') or -1) < right for n in nodes.values()):
                        runs.append([])
                runs[-1].append(c)
        for run_number, run in enumerate(runs):
            for start, end in window_slices(len(run)):
                members = run[start:end]
                payload = {'contract': CONTRACT, 'base_index_id': base['id'],
                           'parse_id': parse_id, 'section_id': section, 'root_run': run_number,
                           'start': start, 'end': end, 'source_chunk_ids': [c['id'] for c in members],
                           'headings': members[0].get('headings', []),
                           'node_ids': list(dict.fromkeys(n for c in members for n in c['node_ids'])),
                           'sources': list({digest(s): s for c in members for s in c['sources']}.values()),
                           'text': '\n\n'.join(c['text'] for c in members)}
                output.append({'id': digest(payload), **payload})
    return output


def summary_key(config, window):
    return digest({'stage': 'body-window-summary-v1', 'window': window,
                   'model': summary_config(config).models.chat.identity(), 'prompt_version': config.enrichment.prompt_version})


def summary_config(config):
    effective = config.model_copy(deep=True)
    effective.models.chat.request_params.update(thinking={'type': 'disabled'}, reasoning_effort='none', max_tokens=1024)
    return effective
