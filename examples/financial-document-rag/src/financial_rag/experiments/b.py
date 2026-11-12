"""B: twenty original coverage entries and the common B/C priority assembler."""
from ..bc_restoration import assemble_restored, select_coverage
from ..common import RagError
from ..structural_expansion import StructuralReader
from .shared import answer


def retrieve(self, query):
    hits = self.searcher.search(query, coverage_selector=lambda hits,k: select_coverage(hits,k,self.catalog,self.index))
    return hits, self.searcher.trace[-1]


def assemble(self, question):
    if (self.config.retrieval.rrf_k != 20 or self.config.retrieval.sparse_k != 50 or
            self.config.retrieval.dense_k != 50 or self.config.retrieval.final_k != 20):
        raise RagError('BC restoration requires frozen RRF20, 50/50 candidates and20 groups')
    hits = self.search(question, bc_coverage=True)
    self.reader = StructuralReader(self.config, self.store, self.original_index)
    self.reader.question = question
    self.assembly = assemble_restored(self.reader, hits, self.index, self.catalog)
    self.trace.extend(self.reader.trace)


def execute(self, question, saved=None, save_initial=None):
    assemble(self, question)
    return answer(self, question)


if __name__ == '__main__':
    from .entry import main
    main('B')
