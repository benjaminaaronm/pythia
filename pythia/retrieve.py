import ollama
from .config import EMBEDDING_MODEL
from .bm25_store import BM25Store


def cosine_similarity(a, b):
    dot_product = sum([x * y for x, y in zip(a, b)])
    norm_a = sum([x ** 2 for x in a]) ** 0.5
    norm_b = sum([x ** 2 for x in b]) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def embed_query(query):
    result = ollama.embed(model=EMBEDDING_MODEL, input=query)
    return result['embeddings'][0]


class RetrieverRanker(object):
    def __init__(self, index_entries):
        self.index = index_entries
        self.bm25_store = BM25Store(index_entries)

    def retrieve(self, query, bm25_n=20, final_n=5):
        bm25_candidates = self.bm25_store.retrieve(query, top_n=bm25_n)
        query_embedding = embed_query(query)
        similarities = []
        for entry, _ in bm25_candidates:
            similarity = cosine_similarity(query_embedding, entry['embedding'])
            similarities.append((entry, similarity))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:final_n]
