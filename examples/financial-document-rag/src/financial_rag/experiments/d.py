"""D: frozen rewrite, initial snapshot, fixed expansion and autonomous source reading."""
import json

from ..agent_reading import AGENT_SYSTEM, TOOLS, FixedReading, original_content, parse_final, response_summary
from ..common import RagError, canonical, digest
from ..context_capacity import Capacity
from ..evidence_packet import restore, snapshot
from ..structural_expansion import StructuralReader, Structure


def execute(self, question, saved=None, save_initial=None):
    if self.index['kind'] != 'structured' or self.original_index['id'] != self.index['id']:
        raise RagError('D may use only the structured original B index')
    self.capacity = Capacity(self.config)
    if saved:
        restore(self.reader, saved)
        self.initial_ranking = saved['packet']['initial_page_ranking']
        self.initial_snapshot = saved
        requested = saved['packet'].get('requested_items', [])
        self.trace.append({'tool':'initial_snapshot_reused','sha256':saved['sha256']})
    else:
        m = self.call([{'role':'system','content':
            'Rewrite the user request into ONE precise retrieval query and a list of requested_items. '
            'Preserve explicitly stated entities, dates, units and scope. Do not invent facts or expected answers. '
            'Return JSON with query and requested_items; no document text or answer is available.'},
            {'role':'user','content':question}], plan=True)
        requested = json.loads(m['content'])
        hits = self.search(requested['query'])
        self.assembly = self.catalog.assemble(self.reader, hits, self.index, expand_neighbors=False)
        self.initial_snapshot = snapshot(self.reader, question, requested['query'], self.initial_ranking, self.assembly)
        self.initial_snapshot['packet']['requested_items'] = requested['requested_items']
        self.initial_snapshot['sha256'] = digest(self.initial_snapshot['packet'])
        requested = requested['requested_items']
        if save_initial:
            save_initial(self.initial_snapshot)
    structure = Structure(self.catalog,self.reader.tokenizer)
    reader = StructuralReader(self.config,self.store,self.original_index,unbounded=True)
    reader.adopt(self.reader, structure)
    self.reader = reader
    self.requested_items = [{'item_id':f'R{i+1}','request':item} for i,item in enumerate(requested or [question])]
    self.reader.unbounded = True
    self.reader.image_limit = self.config.query.agent_total_images
    base_messages=[{'role':'system','content':AGENT_SYSTEM},{'role':'user','content':question}]
    fixed=FixedReading(self.reader,structure,self.capacity.original_batch_tokens(base_messages,TOOLS))
    self.initial_expansion=fixed.read(self.initial_snapshot['packet']['assembly']['anchor_ids'])
    self.trace.append({'tool':'initial_fixed_expansion','actor':'controller','response':self.initial_expansion,
                       'new_evidence':self.initial_expansion['newly_read_ids']})
    messages=base_messages+[{'role':'user','content':[
        {'type':'text','text':'Requested items: '+canonical(self.requested_items)},
        {'type':'text','text':'Automatic first original expansion: '+canonical(response_summary(self.initial_expansion))},
        *original_content(self.reader)]}]
    while self.model_calls < 24:
        m = self.call(messages, tools=TOOLS)
        if not m.get('tool_calls'):
            return parse_final(m['content'])
        messages.append({k:m[k] for k in ('role','content','tool_calls','reasoning_content') if k in m})
        read_ids = set()
        fixed.begin_turn()
        for tc in m['tool_calls']:
            fn = tc["function"]
            args = {}
            before_ids = set(self.reader.evidence)
            self.tool_calls += 1
            try:
                import jsonschema
                args=json.loads(fn.get('arguments','{}'))
                schemas={t['function']['name']:t['function']['parameters'] for t in TOOLS}
                if fn.get("name") not in schemas:
                    raise ValueError(
                        "Unknown tool; use search/read/read_parent/read_source_pages/continue_read/read_evidence"
                    )
                jsonschema.validate(args,schemas[fn['name']])
                if fn['name']=='search':
                    hits = self.search(args["query"])
                    self.tool_calls -= 1
                    response=[{'source_id':h['id'],'kind':h['kind'],'headings':h.get('headings',[]),
                               'sources':h['sources'],'preview':self.reader.tokenizer.prefix(h['text'],self.config.query.search_preview_tokens)} for h in hits]
                elif fn['name']=='read_evidence':
                    eid=args['evidence_id']
                    if eid not in self.reader.evidence:
                        raise RagError("Unknown read evidence_id")
                    response={'results':[{'evidence_id':eid}], 'retained_original':True}
                else:
                    response=getattr(fixed,fn['name'])(**args)
                if isinstance(response,dict):
                    read_ids.update(x['evidence_id'] for x in response.get('results',[]) if x.get('evidence_id'))
            except (RagError,ValueError,TypeError,jsonschema.ValidationError) as error:
                response={'error':str(error).splitlines()[0],
                          'instruction':'Correct the source ID/tool arguments or choose a new query. No original read was fabricated.'}
            detail={'tool':fn.get('name'),'actor':'model','arguments':args,'response':response,
                    'new_evidence':sorted(set(self.reader.evidence)-before_ids),'tool_call_id':tc['id'],'index_id':self.index['id']}
            if fn.get("name") == "search" and isinstance(response, list):
                self.trace[-1].update(detail)
            else:
                self.trace.append(detail)
            messages.append({'role':'tool','tool_call_id':tc['id'],'content':canonical(response_summary(response))})
        if read_ids:
            messages.append({'role':'user','content':original_content(self.reader,read_ids)})
    # Tools requested by exploratory call24 have completed before this boundary.
    messages.append({'role':'user','content':'Exploration has ended. Tools are disabled. Answer now using the originals already read; state unresolved gaps.'})
    final = self.call(messages, forced=True)
    return parse_final(final['content'])


if __name__ == "__main__":
    from .entry import main
    main("D")
