from typing import Callable, Optional
from src.core.metrics.language.levenshtein import Levenshtein
from .base_result import ErrorRateResult

class MatchErrorRate:
    def __init__(self, tokenizer: Optional[Callable] = None):
        self.tokenizer = tokenizer or (lambda x: x.split())

    def calculate(self, reference: str, hypothesis: str) -> ErrorRateResult:
        counters = Levenshtein.compute_with_counter(reference, hypothesis, tokenizer=self.tokenizer)
        S, D, I = counters.substitutions, counters.deletions, counters.insertions
        N = counters.reference_length
        C = N - S - D
        denom = S + I + D + C
        value = (S + I) / denom if denom else 1.0
        return ErrorRateResult(value, I, D, S, C, N)
