from typing import Callable, Optional
from src.core.metrics.language.levenshtein import Levenshtein
from .word_information_preserved import WordInformationPreserved
from .base_result import ErrorRateResult

class WordInformationLost:
    def __init__(self, tokenizer: Optional[Callable] = None):
        self.tokenizer = tokenizer or (lambda x: x.split())

    def calculate(self, reference: str, hypothesis: str) -> ErrorRateResult:
        wip = WordInformationPreserved(self.tokenizer).calculate(reference, hypothesis).value
        value = 1.0 - wip
        return ErrorRateResult(value, 0, 0, 0, 0, 0)
