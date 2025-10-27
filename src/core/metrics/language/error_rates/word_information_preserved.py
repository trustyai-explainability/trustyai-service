from typing import Callable, Optional
from src.core.metrics.language.levenshtein import Levenshtein
from .base_result import ErrorRateResult

class WordInformationPreserved:
    def __init__(self, tokenizer: Optional[Callable] = None):
        self.tokenizer = tokenizer or (lambda x: x.split())

    def calculate(self, reference: str, hypothesis: str) -> ErrorRateResult:
        tokens_ref = self.tokenizer(reference)
        tokens_hyp = self.tokenizer(hypothesis)

        counters = Levenshtein.compute_with_counter(reference, hypothesis, tokenizer=self.tokenizer)
        S, D, I = counters.substitutions, counters.deletions, counters.insertions
        H = counters.reference_length - S - D  # Correct words = reference length - (S + D)

        N_ref = len(tokens_ref)
        N_hyp = len(tokens_hyp)

        # Approximate WIP: (H / N_ref) * (H / N_hyp)
        wip = 0.0
        if N_ref > 0 and N_hyp > 0:
            wip = (H / N_ref) * (H / N_hyp)

        return ErrorRateResult(wip, I, D, S, H, N_ref)