from rank_bm25 import BM25Okapi


def tokenize(text):
    return text.lower().split()


class BM25Store(object):
    def __init__(self, index_entries):
        """
        index_entries: list of dicts with 'text'
        """
        self.index_entries = index_entries
        self.corpus_tokens = [tokenize(e['text']) for e in index_entries]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def retrieve(self, query, top_n=20):
        tokens = tokenize(query)
        query_scores = self.bm25.get_scores(tokens)
        scores = []
        for entry, score in zip(self.index_entries, query_scores):
            scores.append((entry, float(score)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]
