"""B: retrieve structured originals and read their local context."""

from financial_rag.query.bc_restoration import assemble_restored, select_coverage
from financial_rag.query.experiments.shared import answer
from financial_rag.query.structural_expansion import StructuralReader


def retrieve(self, query):
    hits = self.searcher.search(
        query, coverage_selector=lambda hits, k: select_coverage(hits, k, self.catalog, self.index)
    )
    return hits, self.searcher.trace[-1]


def assemble(self, question):
    hits = self.search(question, bc_coverage=True)
    self.reader = StructuralReader(self.config, self.store, self.original_index)
    self.reader.question = question
    self.assembly = assemble_restored(self.reader, hits, self.index, self.catalog)
    self.trace.extend(self.reader.trace)


def execute(self, question, saved=None, save_initial=None):
    assemble(self, question)
    return answer(self, question)
