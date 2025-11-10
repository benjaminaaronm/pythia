import json
import ollama
from .config import EMBEDDING_MODEL, INDEX_PATH


def embed_text(text):
    result = ollama.embed(model=EMBEDDING_MODEL, input=text)
    # Result format: {'embeddings': [[...]]}
    return result['embeddings'][0]


def build_index(chunks):
    """
    chunks: list of {'chunk_id', 'doc_id', 'title', 'text'}
    Returns: list of index entries with embeddings.
    """
    index = []
    for i, ch in enumerate(chunks):
        embedding = embed_text(ch['text'])
        entry = {
            'chunk_id': ch['chunk_id'],
            'doc_id': ch['doc_id'],
            'title': ch['title'],
            'text': ch['text'],
            'embedding': embedding
        }
        index.append(entry)
        if (i+1) % 10 == 0:
            print(f'Embedded {i+1} chunks')
    return index


def save_index(index, path=INDEX_PATH):
    with open(path, 'w', encoding='utf_8') as f:
        json.dump(index, f)
    print(f'Saved index with {len(index)} entries to {path}')


def load_index(path=INDEX_PATH):
    with open(path, 'r', encoding='utf_8') as f:
        index = json.load(f)
    print(f'Loaded index with {len(index)} entries from {path}')
    return index
