import pytest
from src.core.metrics.language.error_rates.word_error_rate import WordErrorRate
from src.core.metrics.language.error_rates.match_error_rate import MatchErrorRate
from src.core.metrics.language.error_rates.word_information_lost import WordInformationLost
from src.core.metrics.language.error_rates.word_information_preserved import WordInformationPreserved

TOLERANCE = 1e-5

def run_test_all(reference_tokens, hypothesis_tokens, expected_wer, expected_mer, expected_wil, expected_wip, tol=TOLERANCE):
    reference = " ".join(reference_tokens)
    hypothesis = " ".join(hypothesis_tokens)

    wer = WordErrorRate().calculate(reference, hypothesis).value
    mer = MatchErrorRate().calculate(reference, hypothesis).value
    wil = WordInformationLost().calculate(reference, hypothesis).value
    wip = WordInformationPreserved().calculate(reference, hypothesis).value

    print(f"[test_all] ref: {reference}\nhyp: {hypothesis}\nWER: {wer} (expected {expected_wer})\nMER: {mer} (expected {expected_mer})\nWIL: {wil} (expected {expected_wil})\nWIP: {wip} (expected {expected_wip})")

    assert pytest.approx(wer, abs=tol) == expected_wer, f"Expected WER: {expected_wer}, got {wer}"
    assert pytest.approx(mer, abs=tol) == expected_mer, f"Expected MER: {expected_mer}, got {mer}"
    assert pytest.approx(wil, abs=tol) == expected_wil, f"Expected WIL: {expected_wil}, got {wil}"
    assert pytest.approx(wip, abs=tol) == expected_wip, f"Expected WIP: {expected_wip}, got {wip}"


def test_equal_reference_hypothesis():
    ref = ["X"]
    hyp = ["X"]
    run_test_all(ref, hyp, expected_wer=0.0, expected_mer=0.0, expected_wil=0.0, expected_wip=1.0)


def test_repeated_hypothesis():
    ref = ["X"]
    hyp = ["X", "X", "Y", "Y"]
    run_test_all(ref, hyp, expected_wer=3.0, expected_mer=0.75, expected_wil=0.75, expected_wip=0.25)


def test_overlap():
    ref = ["X", "Y", "Z"]
    hyp = ["X", "Z"]
    run_test_all(ref, hyp, expected_wer=1/3, expected_mer=1/3, expected_wil=1/3, expected_wip=2/3)