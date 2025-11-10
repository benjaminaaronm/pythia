import ollama
from .config import EMBEDDING_MODEL


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


def retrieve(index, query, top_n=5):
    """
    index: list of entries with keys: 'text', 'embedding', 'doc_id', 'title'
    returns list of (entry, similarity)
    """
    query_embedding = embed_query(query)
    similarities = []
    for entry in index:
        similarity = cosine_similarity(query_embedding, entry['embedding'])
        similarities.append((entry, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]
