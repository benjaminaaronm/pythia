from .vector_store import load_index
from .retrieve import retrieve
from .generate import generate_answer


class RAGPipeline(object):
    def __init__(self, index=None):
        if index is None:
            index = load_index()
        self.index = index

    def answer(self, question, top_n=5):
        retrieved = retrieve(self.index, question, top_n=top_n)
        answer_text = generate_answer(question, retrieved)
        return answer_text, retrieved
