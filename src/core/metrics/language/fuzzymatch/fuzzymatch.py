from typing import List, Optional, Callable
from bleu import BLEU
from levenshtein import WordErrorRate
from difflib import SequenceMatcher


class FuzzyMatch:
    def __init__(self, tokenizer: Optional[Callable[[str], List[str]]] = None):
        """
        :param tokenizer: A callable that tokenizes a string into a list of tokens. Defaults to str.split.
        """
        self.tokenizer = tokenizer or (lambda x: x.split())

    def calculate(self, reference: str, input_str: str) -> bool:
        """
        Compare two strings for exact equality.
        """
        return reference == input_str

    def calculate_wer(self, reference: str, input_str: str, threshold: float) -> bool:
        """
        Return True if the WER between reference and input is less than the threshold.
        """
        wer_metric = WordErrorRate(tokenizer=self.tokenizer)
        wer = wer_metric.calculate(reference, input_str).value
        return wer < threshold

    def calculate_bleu(self, references: List[str], input_str: str, threshold: float) -> bool:
        """
        Return True if the BLEU score is above the threshold.
        """
        bleu_metric = BLEU(tokenizer=self.tokenizer)
        score = bleu_metric.calculate(references, input_str)
        return score > threshold

    def calculate_bleu_ngram(self, references: List[str], input_str: str, threshold: float, max_ngram: int) -> bool:
        """
        BLEU score with custom max n-gram.
        """
        bleu_metric = BLEU(tokenizer=self.tokenizer)
        score = bleu_metric.calculate(references, input_str, max_ngram=max_ngram)
        return score > threshold

    def calculate_bleu_ngram_weights(
        self,
        references: List[str],
        input_str: str,
        threshold: float,
        max_ngram: int,
        weights: List[float]
    ) -> bool:
        """
        BLEU score with custom max n-gram and weights.
        """
        bleu_metric = BLEU(tokenizer=self.tokenizer)
        score = bleu_metric.calculate(references, input_str, max_ngram=max_ngram, weights=weights)
        return score > threshold


     def calculate_similarity(self, reference: str, input_str: str) -> float:
        """
        Return a similarity score between 0 and 1 using difflib's SequenceMatcher.
        """
        return SequenceMatcher(None, reference, input_str).ratio()


    def is_similar(self, reference: str, input_str: str, threshold: float = 0.85) -> bool:
        """
        Return True if the SequenceMatcher similarity exceeds a threshold.
        """
        return self.calculate_similarity(reference, input_str) >= threshold