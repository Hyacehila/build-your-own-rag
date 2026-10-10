"""pre-D: answer an existing, immutable pre-expansion D snapshot only."""
from ..common import RagError
from ..evidence_packet import restore
from .shared import answer


def execute(self, question, saved=None, save_initial=None):
    if saved is None:
        raise RagError('D first evidence snapshot is not available yet')
    restore(self.reader, saved)
    self.initial_ranking = saved['packet']['initial_page_ranking']
    self.initial_snapshot = saved
    self.assembly = saved['packet']['assembly']
    return answer(self, question)


if __name__ == '__main__':
    from .entry import main
    main('pre-D')
