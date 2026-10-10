"""Declared gateway capacity with explicit provenance and counting limitations."""
import copy
from pathlib import Path
import tiktoken
from functools import lru_cache
from .common import RagError, canonical, read_json, sha256


class Capacity:
    def __init__(self, config):
        path = config.query.capacity_evidence_path
        if not path or not Path(path).is_file():
            raise RagError('D needs verified gateway capacity evidence; no guessed default is allowed')
        self.proof = read_json(Path(path))
        p = self.proof
        if (p.get('model') != config.models.chat.model or
            p.get('base_url', '').rstrip('/') != config.models.chat.base_url.rstrip('/') or
            p.get('verification') not in {'operator_configuration', 'provider_capability_metadata', 'provider_error_metadata', 'user_declared'} or
            not p.get('source_reference')):
            raise RagError('Capacity evidence does not attest this gateway/model')
        names = ['context_window_tokens', 'message_overhead_tokens', 'request_overhead_tokens']
        if p.get('image_counting') != 'serialized_payload_tokens_proxy':
            names.append('image_token_upper_bound')
        for name in names:
            if not isinstance(p.get(name), int) or isinstance(p[name], bool) or p[name] <= 0:
                raise RagError('Capacity evidence requires a positive verified ' + name)
        self.encoding = tiktoken.get_encoding(p['text_encoding'])
        self.output = config.models.chat.request_params.get('max_tokens')
        if not isinstance(self.output, int) or self.output <= 0 or self.output >= p['context_window_tokens']:
            raise RagError('Declare a valid provider output reserve')
        self.identity = {'path': str(path), 'sha256': sha256(Path(path)), **p}
        self.limit = p['context_window_tokens']
        self.multiplier = p.get('input_estimate_multiplier', 1.0)
        if not isinstance(self.multiplier, (int,float)) or not 1 <= self.multiplier <= 2:
            raise RagError('Input estimation safety multiplier must be in [1,2]')

    @lru_cache(maxsize=32)
    def text_count(self, text):
        return len(self.encoding.encode(text, disallowed_special=()))

    def count(self, messages, tools):
        images = 0
        def scrub(x):
            nonlocal images
            if isinstance(x, dict):
                if x.get('type') == 'image_url':
                    images += 1
                    if self.proof.get('image_counting') == 'serialized_payload_tokens_proxy':
                        return x
                    return {'type': 'image_url'}
                return {k: scrub(v) for k, v in x.items()}
            return [scrub(v) for v in x] if isinstance(x, list) else x
        # Count image payload strings once per repeated original image, not again for
        # every context-size comparison. Framing reserves are separately declared.
        inline = 0
        def count_payloads(x):
            nonlocal inline
            if isinstance(x, dict):
                if x.get('type') == 'image_url' and self.proof.get('image_counting') == 'serialized_payload_tokens_proxy':
                    inline += self.text_count(canonical(x))
                    return {'type': 'image_url', 'counted_separately': True}
                return {k:count_payloads(v) for k,v in x.items()}
            return [count_payloads(v) for v in x] if isinstance(x,list) else x
        text = canonical({'messages': count_payloads(scrub(messages)), 'tools': tools or []})
        return int((self.text_count(text) + inline +
                images * (0 if self.proof.get('image_counting') == 'serialized_payload_tokens_proxy' else self.proof['image_token_upper_bound']) +
                len(messages) * self.proof['message_overhead_tokens'] +
                self.proof['request_overhead_tokens']) * self.multiplier) + self.output

    def original_batch_tokens(self, base_messages, tools):
        # Capacity-derived page size, not a cumulative original-reading budget.
        # Half of the available space accommodates source identities, tool results,
        # existing context and tokenizer mismatch; the rest is rereadable via cursor.
        return max(1, int((self.limit-self.count(base_messages,tools))/(2*self.multiplier)))

    def after_rejection(self, messages, tools, trace, diagnostics):
        maximum=diagnostics.get('context_window_tokens')
        if isinstance(maximum,int) and maximum>self.output:
            self.limit=min(self.limit,maximum)
        trace.append({'tool':'capacity_rejection','diagnostics':diagnostics,
                      'action':'one_compacted_physical_retry','context_window_tokens':self.limit})
        return self.fit(messages,tools,trace,force=True)

    def fit(self, messages, tools, trace, force=False):
        packed = copy.deepcopy(messages)
        # More than eight inline images are represented by explicit retained-source pointers.
        image_parts = [(i, j) for i, m in enumerate(packed) if isinstance(m.get('content'), list)
                       for j, part in enumerate(m['content']) if part.get('type') == 'image_url']
        image_pointers = 0
        for i, j in image_parts[:-8]:
            image_pointers += 1
            packed[i]['content'][j] = {'type': 'text', 'text': '[Earlier original image retained in evidence store; read its source again to display.]'}
        # For an explicitly user-configured capacity with unknown image token rules,
        # count serialized image payloads as a transparent proxy, not provider billing.
        # Keep newest images that fit; full originals remain redisplayable in the vault.
        for i, j in image_parts[-8:]:
            if not force and self.count(packed, tools) <= self.limit:
                break
            image_pointers += 1
            packed[i]['content'][j] = {'type': 'text', 'text': '[Original image retained in evidence store; omitted from this request under the declared capacity proxy.]'}
        compacted = []
        # Never compact system, original question, or latest instruction/tool result.
        for i in range(2, max(2, len(packed)-1)):
            if not force and self.count(packed, tools) <= self.limit:
                break
            m = packed[i]
            if m['role'] in {'tool', 'user'} and m.get('content'):
                import re
                text = canonical(m['content'])
                ids = sorted(set(re.findall(r'\bE\d+\b', text)))
                m['content'] = 'Earlier tool content remains in the evidence store. Source IDs: ' + ', '.join(ids) + '. Use read_evidence to reread originals; this pointer is not evidence.'
                compacted.append(i)
        count = self.count(packed, tools)
        trace.append({'tool': 'context_capacity', 'count_including_output_reserve': count,
                      'context_window_tokens': self.limit,
                      'compacted_message_indices': compacted,
                      'image_pointers': image_pointers,
                      'capacity_verification': self.proof['verification'],
                      'counting_is_estimated': self.proof.get('counting_is_estimated', False)})
        if count > self.limit:
            raise RagError('Request cannot fit the declared gateway capacity without losing its current input')
        return packed
