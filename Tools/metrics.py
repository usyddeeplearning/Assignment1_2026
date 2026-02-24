import random
import re
import string
from collections import Counter
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from Losses import qa_nll_loss


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    return max(metric_fn(prediction, gt) for gt in ground_truths)


def evaluate(eval_file, answer_dict):
    f1 = exact_match = total = 0.0
    for key, pred in answer_dict.items():
        total += 1.0
        ground_truths = eval_file[key]["answers"]
        exact_match += metric_max_over_ground_truths(exact_match_score, pred, ground_truths)
        f1 += metric_max_over_ground_truths(f1_score, pred, ground_truths)
    return {"exact_match": 100.0 * exact_match / total, "f1": 100.0 * f1 / total}


def convert_tokens(eval_file, qa_id, pp1, pp2):
    answer_dict = {}
    remapped_dict = {}
    for qid, p1, p2 in zip(qa_id, pp1, pp2):
        context = eval_file[str(qid)]["context"]
        spans = eval_file[str(qid)]["spans"]
        uuid = eval_file[str(qid)]["uuid"]
        length = len(spans)
        if p1 >= length or p2 >= length:
            ans = ""
        else:
            start_idx = spans[p1][0]
            end_idx = spans[p2][1]
            ans = context[start_idx:end_idx]
        answer_dict[str(qid)] = ans
        remapped_dict[uuid] = ans
    return answer_dict, remapped_dict


@torch.no_grad()
def run_eval(model, dataset, eval_file, num_batches, use_random_batches, ema: Optional[object], use_ema: bool, device, loss_fn=qa_nll_loss):
    model.eval()
    answer_dict = {}
    losses = []

    if use_random_batches:
        steps = random.sample(range(0, len(dataset)), min(num_batches, len(dataset)))
    else:
        steps = list(range(min(num_batches, len(dataset))))

    if ema is not None and use_ema:
        ema.apply(model)

    for step in tqdm(steps, total=len(steps)):
        Cwid, Ccid, Qwid, Qcid, y1, y2, ids = dataset[step]
        Cwid, Ccid, Qwid, Qcid = Cwid.to(device), Ccid.to(device), Qwid.to(device), Qcid.to(device)
        y1, y2 = y1.to(device), y2.to(device)

        p1, p2 = model(Cwid, Ccid, Qwid, Qcid)
        loss = loss_fn(p1, p2, y1, y2)
        losses.append(float(loss.item()))

        yp1 = torch.argmax(p1, dim=1)
        yp2 = torch.argmax(p2, dim=1)
        yps = torch.stack([yp1, yp2], dim=1)
        ymin, _ = torch.min(yps, dim=1)
        ymax, _ = torch.max(yps, dim=1)

        answer_dict_, _ = convert_tokens(eval_file, ids.tolist(), ymin.tolist(), ymax.tolist())
        answer_dict.update(answer_dict_)

    if ema is not None and use_ema:
        ema.restore(model)

    metrics = evaluate(eval_file, answer_dict)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics, answer_dict
