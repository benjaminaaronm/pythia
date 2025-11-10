import json
import re
from .rag import RAGPipeline


def normalize(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def f1_score(prediction, ground_truth):
    pred_tokens = normalize(prediction).split()
    gt_tokens = normalize(ground_truth).split()

    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0

    common = set(pred_tokens) & set(gt_tokens)
    num_same = 0
    for token in common:
        num_same += min(pred_tokens.count(token), gt_tokens.count(token))

    if num_same == 0:
        return 0.0

    precision = float(num_same) / len(pred_tokens)
    recall = float(num_same) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction, ground_truth):
    return normalize(prediction) == normalize(ground_truth)


def load_eval_set(path):
    with open(path, 'r', encoding='utf_8') as f:
        return json.load(f)


def evaluate(path_to_eval_set):
    eval_set = load_eval_set(path_to_eval_set)
    pipeline = RAGPipeline()

    total_f1 = 0.0
    total_em = 0.0
    n = len(eval_set)

    for ex in eval_set:
        q = ex['question']
        gt = ex['answer']
        print('\nQuestion:', q)
        pred, _ = pipeline.answer(q)
        print('Prediction:', pred)
        print('Ground truth:', gt)

        f1 = f1_score(pred, gt)
        em = 1.0 if exact_match(pred, gt) else 0.0
        print(f'F1: {f1:.3f}, EM: {em:.3f}')

        total_f1 += f1
        total_em += em

    avg_f1 = total_f1 / n if n > 0 else 0.0
    avg_em = total_em / n if n > 0 else 0.0

    print('\n=== Overall results ===')
    print(f'Avg F1: {avg_f1:.3f}')
    print(f'Exact Match: {avg_em:.3f}')


if __name__ == '__main__':
    from os import path
    from .config import BASE_DIR

    eval_path = path.join(BASE_DIR, 'data', 'eval', 'eval_set.json')
    evaluate(eval_path)
