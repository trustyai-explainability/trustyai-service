from typing import Callable, List, Union
from dataclasses import dataclass
from nltk.metrics.distance import edit_distance, edit_distance_align
from src.core.metrics.language.utils import clean_text



@dataclass
class LevenshteinResult:
    distance: int
    insertions: int
    deletions: int
    substitutions: int
    reference_length: int

    def normalized_distance(self) -> float:
        if self.reference_length == 0:
            return 0.0
        return self.distance / self.reference_length

class Levenshtein:

    @staticmethod
    def compute_(
        reference: str,
        hypothesis: str,
        tokenizer: Callable[[str], List[str]] = None
    ) -> LevenshteinResult:
        """
        Compute Levenshtein distance at the character or token level.

        :param reference: Ground truth string.
        :param hypothesis: Predicted string.
        :param tokenizer: Optional function to split input into tokens. If None, character-level is used.
        """

        clean_ref = clean_text(reference)
        clean_hyp = clean_text(hypothesis)

        return edit_distance(clean_ref, clean_hyp)


    @staticmethod
    def compute_with_counter(
        reference: str,
        hypothesis: str,
        tokenizer: Callable[[str], List[str]] = None
    ) -> LevenshteinResult:
        """
        Compute Levenshtein distance at the character or token level.

        :param reference: Ground truth string.
        :param hypothesis: Predicted string.
        :param tokenizer: Optional function to split input into tokens. If None, character-level is used.
        """
        clean_ref = clean_text(reference)
        clean_hyp = clean_text(hypothesis)

        if tokenizer:
            ref_seq = tokenizer(clean_ref)
            hyp_seq = tokenizer(clean_hyp)
        else:
            ref_seq = list(clean_ref)
            hyp_seq = list(clean_hyp)

        alignment = edit_distance_align(ref_seq, hyp_seq)

        aligned_ref = []
        aligned_hyp = []
        for i, j in alignment:
            aligned_ref.append(ref_seq[i] if i < len(ref_seq) else '*')
            aligned_hyp.append(hyp_seq[j] if j < len(hyp_seq) else '*')


        insertions = deletions = substitutions = 0
        for r, h in zip(aligned_ref, aligned_hyp):
            if r == h:
                continue
            elif r == '*':
                insertions += 1
            elif h == '*':
                deletions += 1
            else:
                substitutions += 1

        total_distance = insertions + deletions + substitutions
        return LevenshteinResult(
            distance=total_distance,
            insertions=insertions,
            deletions=deletions,
            substitutions=substitutions,
            reference_length=len(ref_seq)
        )