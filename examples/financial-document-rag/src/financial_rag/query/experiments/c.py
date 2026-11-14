"""C: protected B originals followed by at most five novel summary entries."""

from financial_rag.query.bc_restoration import select_coverage, select_supplement
from financial_rag.query.experiments.b import assemble
from financial_rag.query.experiments.shared import answer
from financial_rag.query.retrieval import Search, page_ranking


def retrieve(self, query):
    base = self.original_searcher or Search(self.config, self.store, self.original_index, self.api)
    base.api = self.api
    primary = base.search(
        query, coverage_selector=lambda h, k: select_coverage(h, k, self.catalog, self.original_index)
    )
    primary_trace = base.trace[-1]
    limit = self.config.retrieval.summary_supplement_k
    if limit:
        extra = self.searcher.search(
            query,
            final_k=limit,
            coverage_selector=lambda h, k: select_supplement(h, k, self.catalog, self.index, primary),
        )
        summary_trace = dict(self.searcher.trace[-1])
    else:
        extra = []
        summary_trace = {
            "tool": "search",
            "query": query,
            "index_id": self.index["id"],
            "sparse": [],
            "dense": [],
            "coverage_selection": [],
            "rrf_k": self.config.retrieval.rrf_k,
            "final_k": 0,
        }
    hits = primary + extra
    trace = {
        **summary_trace,
        "original_search": primary_trace,
        "retrieval_contract": "originals_then_summary-v1",
        "protected_original_ids": [h["id"] for h in primary],
        "summary_supplement_ids": [h["id"] for h in extra],
        "ranked_chunks": [{"id": h["id"], "score": h["score"]} for h in hits],
        "source_groups_returned": len(hits),
        "page_ranking": page_ranking(hits),
        "same_query_index_lookups": 2 if limit else 1,
    }
    return hits, trace


def execute(self, question, saved=None, save_initial=None):
    assemble(self, question)
    return answer(self, question)
