# evaluation metrics for LLM
from typing import List
import torch
from nltk.translate.bleu_score import sentence_bleu


def bleu_score(pred : str, label : str) -> float:
    """
    :param pred:
    :param label:
    :return:
    """
    return sentence_bleu(label, pred)
