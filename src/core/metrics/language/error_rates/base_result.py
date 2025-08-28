from dataclasses import dataclass

@dataclass
class ErrorRateResult:
    value: float
    insertions: int
    deletions: int
    substitutions: int
    correct: int
    reference_length: int
