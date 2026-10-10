"""Shared one-call answer protocol for A/B/C/pre-D."""

def answer(self, question):
    from ..runner import SYSTEM, parse_answer
    message = self.call([{'role':'system','content':SYSTEM},
                         {'role':'user','content':[{'type':'text','text':question}, *self.reader.content()]}])
    return parse_answer(message['content'])
