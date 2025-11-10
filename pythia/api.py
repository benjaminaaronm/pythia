from fastapi import FastAPI
from pydantic import BaseModel
from .rag import RAGPipeline


app = FastAPI(title='pythia RAG API')

pipeline = RAGPipeline()


class Query(BaseModel):
    question: str
    top_n: int = 5


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/query')
def query(q: Query):
    answer, retrieved = pipeline.answer(q.question, final_n=q.top_n)

    contexts = []
    for entry, score in retrieved:
        contexts.append({
            'doc_id': entry['doc_id'],
            'title': entry['title'],
            'score': score,
            'text': entry['text']
        })

    return {
        'question': q.question,
        'answer': answer,
        'contexts': contexts
    }
