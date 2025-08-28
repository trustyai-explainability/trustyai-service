import pytest
from src.core.metrics.language.bleu import BLEUMetric as BLEU
from nltk.translate.bleu_score import SmoothingFunction


common_hypothesis = "the cat the cat on mat"
common = "the cat is on the mat"
uncommon = "The candidate has no alignment to any of the references"
common_references = [common, common]

validation_reference = [
    "It is a guide to action that ensures that the military will forever heed Party commands",
    "It is the guiding principle which guarantees the military forces always being under the command of the Party",
    "It is the practical guide for the army always to heed the directions of the party"
]

validation_hypothesis_a = "It is a guide to action which ensures that the military always obeys the commands of the party"
validation_hypothesis_b = "It is to insure the troops forever hearing the activity guidebook that party direct"

hyp1 = "It is a guide to action which ensures that the military always obeys the commands of the party"
hyp2 = "he read the book because he was interested in world history"

ref1a = "It is a guide to action that ensures that the military will forever heed Party commands"
ref1b = "It is the guiding principle which guarantees the military forces always being under the command of the Party"
ref1c = "It is the practical guide for the army always to heed the directions of the party"
ref2a = "he was interested in world history because he read the book"

def test_sentence_epsilon_smoothing_weights():
    smoothing = SmoothingFunction().method1
    bleu = BLEU(smoothing_method=smoothing)
    score = bleu.calculate(common_references, common_hypothesis, max_ngram=2, weights=[0.3, 0.7])
    assert pytest.approx(score, 0.05) == 0.3

def test_sentence_no_smoothing_weights():
    bleu = BLEU()
    score = bleu.calculate(common_references, common_hypothesis, max_ngram=2, weights=[0.3, 0.7])
    assert pytest.approx(score, 0.05) == 0.3

def test_sentence_no_smoothing_no_weights():
    bleu = BLEU()
    score = bleu.calculate(common_references, common_hypothesis, max_ngram=2)
    assert pytest.approx(score, abs=0.01) == 0.4082

def test_zero_matches():
    references = [uncommon]
    hypothesis = "John loves Mary"
    bleu = BLEU()
    for n in range(1, 6):
        assert bleu.calculate(references, hypothesis, max_ngram=n) == 0.0

def test_full_matches():
    references = [uncommon] * 4
    bleu = BLEU()
    for n in range(1, 11):
        assert bleu.calculate(references, uncommon, max_ngram=n) == 1.0



def test_validation_bleu_2():
    bleu = BLEU()
    assert pytest.approx(bleu.calculate(validation_reference, validation_hypothesis_a, max_ngram=2), 0.05) == 0.7453

def test_validation_bleu_3():
    bleu = BLEU()
    assert pytest.approx(bleu.calculate(validation_reference, validation_hypothesis_a, max_ngram=3), 0.05) == 0.6240

def test_validation_bleu_4():
    bleu = BLEU()
    assert pytest.approx(bleu.calculate(validation_reference, validation_hypothesis_a, max_ngram=4), 0.02) == 0.5045

def test_validation_bleu_5():
    bleu = BLEU()
    assert pytest.approx(bleu.calculate(validation_reference, validation_hypothesis_a, max_ngram=5), 0.02) == 0.3920

def test_bleu_corpus():
    bleu = BLEU()
    references = [[ref1a, ref1b, ref1c], [ref2a]]
    hypotheses = [hyp1, hyp2]
    weights = [0.25] * 4
    score = bleu.calculate_corpus(references, hypotheses, max_ngram=4, weights=weights)
    assert pytest.approx(score, 0.01) == 0.5920

def test_bleu_corpus_individual_consistency():
    bleu = BLEU()
    score1 = bleu.calculate([ref1a, ref1b, ref1c], hyp1)
    score2 = bleu.calculate([ref2a], hyp2)
    average = (score1 + score2) / 2.0
    assert pytest.approx(average, 0.05) == 0.6223

def test_bleu_corpus_custom_weights():
    bleu = BLEU()
    weights = [0.1, 0.3, 0.5, 0.1]
    references = [[ref1a, ref1b, ref1c], [ref2a]]
    hypotheses = [hyp1, hyp2]
    assert pytest.approx(bleu.calculate_corpus(references, hypotheses, max_ngram=4, weights=weights), 0.01) == 0.5818

def test_bleu_corpus_multiple_weight_sets():
    bleu = BLEU()
    weight_sets = [
        [0.5, 0.5],
        [0.333, 0.333, 0.334],
        [0.25, 0.25, 0.25, 0.25],
        [0.2, 0.2, 0.2, 0.2, 0.2]
    ]
    expected_scores = [0.8242, 0.7067, 0.5920, 0.4719]
    references = [[ref1a, ref1b, ref1c], [ref2a]]
    hypotheses = [hyp1, hyp2]

    for weights, expected in zip(weight_sets, expected_scores):
        ngram = len(weights)
        score = bleu.calculate_corpus(references, hypotheses, max_ngram=ngram, weights=weights)
        assert pytest.approx(score, 0.02) == expected