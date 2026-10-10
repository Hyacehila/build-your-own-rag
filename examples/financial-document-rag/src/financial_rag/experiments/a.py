"""A: the current flat ten-entry baseline; no structural expansion."""
from .shared import answer


def execute(self, question, saved=None, save_initial=None):
    hits = self.search(question)
    self.assembly = self.catalog.assemble(self.reader, hits, self.index, expand=False)
    return answer(self, question)


if __name__ == '__main__':
    from .entry import main
    main('A')
