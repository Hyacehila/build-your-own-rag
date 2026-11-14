"""A: the current flat ten-entry baseline; no structural expansion."""

from financial_rag.query.experiments.shared import answer


def execute(self, question, saved=None, save_initial=None):
    hits = self.search(question)
    self.assembly = self.catalog.assemble(self.reader, hits, self.index, expand=False)
    return answer(self, question)
