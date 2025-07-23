import re
from rouge_score import rouge_scorer
from nltk.tokenize import sent_tokenize, word_tokenize
from typing import Literal, List
from utils import clean_text

class ROUGEMetric:
    def __init__(self, rouge_type: Literal["rouge1", "rouge2", "rougeL", "rougeLsum"] = "rougeL"):
        """
        :param rouge_type: The type of ROUGE score to compute.
                           Supported values:
                           - "rouge1" : unigram overlap
                           - "rouge2" : bigram overlap
                           - "rougeL" : longest common subsequence (LCS)
                           - "rougeLsum" : sentence-level LCS averaged over pairs
        """
        self.rouge_type = rouge_type
        self.scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)

    def calculate(self, reference: str, hypothesis: str) -> float:
        """
        Calculates the ROUGE score between a single reference and hypothesis.

        : param reference: The ground truth string.
        : param hypothesis: The generated string.
        : return: The ROUGE F1 score as a float.
        """
        if self.rouge_type == "rougeLsum":
            return self._rouge_lsum(reference, hypothesis)
        else:
            reference = clean_text(reference)
            hypothesis = clean_text(hypothesis)
            score = self.scorer.score(reference, hypothesis)
            return score[self.rouge_type].fmeasure

    def _rouge_lsum(self, reference: str, hypothesis: str) -> float:
        """
        Calculates the ROUGE-Lsum score by averaging sentence-level ROUGE-L scores.

        :param reference: A reference paragraph consisting of multiple sentences.
        :param hypothesis: A hypothesis paragraph consisting of multiple sentences.
        :return: The average ROUGE-L F1 score over aligned sentence pairs.
        """
        reference = clean_text(reference)
        hypothesis = clean_text(hypothesis)
        ref_sents = sent_tokenize(reference)
        hyp_sents = sent_tokenize(hypothesis)

        total_score = 0.0
        count = min(len(ref_sents), len(hyp_sents))
        for i in range(count):
            score = self.scorer.score(ref_sents[i], hyp_sents[i])
            total_score += score["rougeLsum"].fmeasure

        return total_score / count if count > 0 else 0.0