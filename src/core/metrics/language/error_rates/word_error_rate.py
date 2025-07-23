from typing import Callable, Optional
from levenshtein import Levenshtein
from .base_result import ErrorRateResult

class WordErrorRate:
    def __init__(self, tokenizer: Optional[Callable] = None):
        self.tokenizer = tokenizer or (lambda x: x.split())

    def calculate(self, reference: str, hypothesis: str) -> ErrorRateResult:
        counters = Levenshtein.compute_with_counter(reference, hypothesis, tokenizer=self.tokenizer)
        S, D, I = counters.substitutions, counters.deletions, counters.insertions
        N = counters.reference_length
        value = (S + D + I) / N if N else 1.0
        return ErrorRateResult(value, I, D, S, 0, N)
