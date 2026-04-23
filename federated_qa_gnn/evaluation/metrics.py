from __future__ import annotations

import logging
from typing import List

import torch

logger = logging.getLogger("federated_qa")


def compute_rouge_l(predictions: List[str], references: List[str]) -> float:
    """
    Compute corpus-level ROUGE-L F1 using the rouge_score library.

    Args:
        predictions: list of model-generated strings
        references:  list of ground-truth strings (same length)

    Returns:
        Mean ROUGE-L F1 score across all pairs.
    """
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    total = 0.0
    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        total += score["rougeL"].fmeasure
    return total / max(len(predictions), 1)


def compute_bleu4(predictions: List[str], references: List[str]) -> float:
    """
    Compute corpus-level BLEU-4 with smoothing (method1).

    Args:
        predictions: list of model-generated strings
        references:  list of ground-truth strings

    Returns:
        Smoothed BLEU-4 score.
    """
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    smoothie = SmoothingFunction().method1
    total = 0.0
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.split()
        ref_tokens = [ref.split()]
        score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothie)
        total += score
    return total / max(len(predictions), 1)


def compute_bertscore(
    predictions: List[str],
    references: List[str],
    device: torch.device,
) -> float:
    """
    Compute mean BERTScore F1 using distilbert-base-uncased.

    Args:
        predictions: list of model-generated strings
        references:  list of ground-truth strings
        device:      torch device for BERTScore model

    Returns:
        Mean BERTScore F1.
    """
    from bert_score import score as bert_score

    _, _, F1 = bert_score(
        predictions,
        references,
        model_type="distilbert-base-uncased",
        device=str(device),
        verbose=False,
    )
    return F1.mean().item()


def compute_all_metrics(
    predictions: List[str],
    references: List[str],
    device: torch.device,
) -> dict[str, float]:
    """
    Compute ROUGE-L, BLEU-4, and BERTScore F1 in one call.

    Returns:
        dict with keys 'rouge_l', 'bleu_4', 'bertscore_f1'
    """
    if not predictions:
        return {"rouge_l": 0.0, "bleu_4": 0.0, "bertscore_f1": 0.0}

    rouge = compute_rouge_l(predictions, references)
    bleu = compute_bleu4(predictions, references)
    bert = compute_bertscore(predictions, references, device)
    return {"rouge_l": rouge, "bleu_4": bleu, "bertscore_f1": bert}
