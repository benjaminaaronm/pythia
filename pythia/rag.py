from .vector_store import load_index
from .retrieve import RetrieverRanker
from .generate import generate_answer


class RAGPipeline(object):
    def __init__(self, index=None):
        if index is None:
            index = load_index()
        self.index = index
        self.retriever_ranker = RetrieverRanker(index)

    def answer(self, question, final_n=5):
        retrieved = self.retriever_ranker.retrieve(question, final_n=final_n)
        answer_text = generate_answer(question, retrieved)
        return answer_text, retrieved
