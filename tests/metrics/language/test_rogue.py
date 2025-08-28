import pytest
from src.core.metrics.language.rogue import ROUGEMetric as ROUGE

def test_rouge1():
    rouge = ROUGE(rouge_type="rouge1")
    score = rouge.calculate("testing one two", "testing")
    assert score == 0.5

def test_rouge_scores_empty():
    for rouge_type in ["rouge1", "rouge2", "rougeL", "rougeLsum"]:
        rouge = ROUGE(rouge_type=rouge_type)
        score = rouge.calculate("testing one two", "")
        assert score == 0

def test_rouge2():
    rouge = ROUGE(rouge_type="rouge2")
    score = rouge.calculate("testing one two", "testing one")
    assert pytest.approx(score, abs=0.05) == 0.66

def test_rougel_consecutive():
    rouge = ROUGE(rouge_type="rougeL")
    score = rouge.calculate("testing one two", "testing one")
    assert pytest.approx(score, abs=0.05) == 0.8

def test_rougel_non_consecutive():
    rouge = ROUGE(rouge_type="rougeL")
    score = rouge.calculate("testing one two", "testing two")
    assert pytest.approx(score, abs=0.05) == 0.8

def test_rougel_sum():
    rouge = ROUGE(rouge_type="rougeLsum")
    score = rouge.calculate("w1 w2 w3 w4 w5", "w1 w2 w6 w7 w8\nw1 w3 w8 w9 w5")
    assert pytest.approx(score, abs=0.05) == 0.5

def test_rougel_sum_non_word():
    rouge = ROUGE(rouge_type="rougeLsum")
    score = rouge.calculate("w1 w2 w3 w4 w5", "/")
    assert score == 0