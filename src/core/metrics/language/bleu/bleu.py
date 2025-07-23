from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from typing import List, Optional, Callable


class BLEUMetric:
    def __init__(
        self,
        smoothing_method: Optional[int] = None,
        tokenizer: Optional[Callable[[str], List[str]]] = None
    ):
        """
        :param smoothing_method: Smoothing method number (1-7) from nltk's SmoothingFunction.
                                 None means no smoothing.
        :param tokenizer: A callable that tokenizes a string into a list of tokens. Defaults to str.split.
        """
        self.smoothing_function = (
            getattr(SmoothingFunction(), f"method{smoothing_method}")
            if smoothing_method is not None else None
        )
        self.tokenizer = tokenizer or (lambda x: x.split())


    @staticmethod
    def create_uniform_weights(max_ngram: int) -> List[float]:
        """
        Create uniform weights for BLEU-N scoring.
        """
        return [1.0 / max_ngram] * max_ngram

    def calculate(self, references: List[str], hypothesis: str, max_ngram: int = 4, weights: Optional[List[float]] = None) -> float:
        """
        Calculate sentence-level BLEU score.
        """
        if weights is None:
            weights = self.create_uniform_weights(max_ngram)

        tokenized_refs = [word_tokenize(ref) for ref in references]
        tokenized_hyp = word_tokenize(hypothesis)

        return sentence_bleu(
            tokenized_refs,
            tokenized_hyp,
            weights=weights,
            smoothing_function=self.smoothing_function
        )

    def calculate_corpus(self, references: List[List[str]], hypotheses: List[str], max_ngram: int = 4, weights: Optional[List[float]] = None) -> float:
        """Calculate corpus-level BLEU score.

        :param references: List of lists of reference strings. One list of references per hypothesis.
        :param hypotheses: List of hypothesis strings.
        """
        if weights is None:
            weights = self.create_uniform_weights(max_ngram)

        tokenized_refs = [[word_tokenize(ref) for ref in ref_group] for ref_group in references]
        tokenized_hyps = [word_tokenize(hyp) for hyp in hypotheses]

        return corpus_bleu(
            tokenized_refs,
            tokenized_hyps,
            weights=weights,
            smoothing_function=self.smoothing_function
        )