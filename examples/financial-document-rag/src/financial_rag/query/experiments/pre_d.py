"""pre-D: answer an existing, immutable pre-expansion D snapshot only."""

from financial_rag.common import RagError
from financial_rag.query.evidence_packet import restore
from financial_rag.query.experiments.shared import answer


def execute(self, question, saved=None, save_initial=None):
    if saved is None:
        raise RagError("D first evidence snapshot is not available yet")
    restore(self.reader, saved)
    self.initial_ranking = saved["packet"]["initial_page_ranking"]
    self.initial_snapshot = saved
    self.assembly = saved["packet"]["assembly"]
    return answer(self, question)
