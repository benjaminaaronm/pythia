from .ingest import load_documents, chunk_documents
from .vector_store import build_index, save_index


def main():
    print('Loading documents...')
    docs = load_documents(subdir='wikipedia')
    print(f'Loaded {len(docs)} docs')

    print('Chunking documents...')
    chunks = chunk_documents(docs)
    print(f'Created {len(chunks)} chunks')

    print('Building index (embedding)...')
    index = build_index(chunks)

    print('Saving index...')
    save_index(index)
    print('Done.')


if __name__ == '__main__':
    main()
