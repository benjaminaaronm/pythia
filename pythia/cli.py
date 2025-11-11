from .rag import RAGPipeline


def main():
    pipeline = RAGPipeline()

    print('pythia - simple RAG CLI. Ctrl+C to quit.')
    while True:
        try:
            question = input('\nQuestion: ')
        except (EOFError, KeyboardInterrupt):
            print('\nBye.')
            break

        if not question.strip():
            continue

        answer, retrieved = pipeline.answer(question)
        print(f'\nAnswer:\n{answer}\n')

        print('---- Retrieved context ----')
        for entry, score in retrieved:
            preview = entry["text"][:120].replace("\n", " ") + "..."
            print(f'[{score:.2f}] {entry["doc_id"]} - {preview}')


if __name__ == '__main__':
    main()
