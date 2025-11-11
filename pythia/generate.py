import ollama
from .config import LL_MODEL


def build_prompt(question, context_entries):
    context_parts = []
    for entry, score in context_entries:
        part = f'Source: {entry["title"]} (doc_id={entry["doc_id"]}, score={score:.2f})\n{entry["text"]}'
        context_parts.append(part)

    context_text = '\n\n'.join(context_parts)

    system_prompt = f"""You are pythia, a helpful assistant.
Use ONLY the following context to answer the user question.
If the answer is not in the context, say you don't know.

Context:
{context_text}
"""

    return system_prompt


def generate_answer(question, context_entries):
    system_prompt = build_prompt(question, context_entries)

    response = ollama.chat(
        model=LL_MODEL,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': question},
        ],
        stream=False
    )

    return response['message']['content']
