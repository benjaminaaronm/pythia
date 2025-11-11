import os
import unicodedata
from .config import RAW_DATA_DIR
from .utils import list_text_files


def load_documents(subdir='wikipedia'):
    """
    Load all .txt files from data/raw/<subdir>.
    Returns a list of dicts: {'doc_id': str, 'title': str, 'text': str, 'path': str}
    """
    directory = os.path.join(RAW_DATA_DIR, subdir)
    docs = []

    if not os.path.exists(directory):
        raise ValueError(f'Directory does not exist: {directory}')

    for path in list_text_files(directory):
        with open(path, 'r', encoding='utf_8') as f:
            text = unicodedata.normalize('NFKD', f.read()).replace("  ", " ").replace("\n", " ").strip()
        title = os.path.splitext(os.path.basename(path))[0]
        doc_id = title
        docs.append({
            'doc_id': doc_id,
            'title': title,
            'text': text,
            'path': path
        })
    return docs


def chunk_text(text, max_chars=800, overlap_chars=100):
    """
    Very simple character-based chunking with overlap.
    """
    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + max_chars, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap_chars
        if end == length:
            break

    return chunks


def chunk_documents(docs, max_chars=800, overlap_chars=100):
    """
    Given a list of docs, return a list of chunk dicts:
    {'chunk_id': int, 'doc_id': str, 'text': str, 'title': str}
    """
    chunks = []
    chunk_id = 0
    for doc in docs:
        doc_chunks = chunk_text(doc['text'], max_chars=max_chars, overlap_chars=overlap_chars)
        for ch in doc_chunks:
            chunks.append({
                'chunk_id': chunk_id,
                'doc_id': doc['doc_id'],
                'title': doc['title'],
                'text': ch
            })
            chunk_id += 1
    return chunks


if __name__ == '__main__':
    test_text = load_documents()[0]["text"]
    print(test_text)
    test_text = """
                ABCDEFGHIJKLMNOPQRSTUVWXYZ
                """
    print(chunk_text(test_text.strip(), 5, 2))
