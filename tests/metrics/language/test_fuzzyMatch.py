import pytest
from src.core.metrics.language.match.fuzzymatch import FuzzyMatch


references = [
    "This is the test reference, to which I will compare alignment against.",
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Curabitur condimentum velit id velit posuere dictum. Fusce euismod tortor massa, nec euismod sapien laoreet non. Donec vulputate mi velit, eu ultricies nibh iaculis vel. Aenean posuere urna nec sapien consectetur, vitae porttitor sapien finibus. Duis nec libero convallis lectus pharetra blandit ut ac odio. Vivamus nec dui quis sem convallis pulvinar. Maecenas sodales sollicitudin leo a faucibus.",
    "The quick red fox jumped over the lazy brown dog"
]

inputs = [
    "I'm a hypothesis reference, from which the aligner  will compare against.",
    "Lorem ipsum sit amet, consectetur adipiscing elit. Curabitur condimentum velit id velit posuere dictum. Fusce blandit euismod tortor massa, nec euismod sapien blandit laoreet non. Donec vulputate mi velit, eu ultricies nibh iaculis vel. Aenean posuere urna nec sapien consectetur, vitae porttitor sapien finibus. Duis nec libero convallis lectus pharetra blandit ut ac odio. Vivamus nec dui quis sem convallis pulvinar. Maecenas sodales sollicitudin leo a faucibus.",
    "The quick red fox jumped over the lazy brown dog"
]


def commons_tokenizer(text: str):
    return [token for token in text.split() if token.strip()]


def test_exact_match():
    expected = [False, False, True]
    fm = FuzzyMatch()
    for i, (ref, hyp) in enumerate(zip(references, inputs)):
        assert fm.calculate(ref, hyp) == expected[i]

def test_wer_match_default_tokenizer():
    expected = [True, True, True]
    fm = FuzzyMatch()
    for i, (ref, hyp) in enumerate(zip(references, inputs)):
        actual = fm.calculate_wer(ref, hyp, threshold=0.67) # 0.65 is too low for this test
        assert actual == expected[i]

def test_wer_match_commons_tokenizer():
    expected = [False, True, True]
    fm = FuzzyMatch(tokenizer=commons_tokenizer)
    for i, (ref, hyp) in enumerate(zip(references, inputs)):
        assert fm.calculate_wer(ref, hyp, threshold=0.65) == expected[i]

def test_bleu_match_default_tokenizer():
    expected = [False, True, True]
    fm = FuzzyMatch()
    for i, (ref, hyp) in enumerate(zip(references, inputs)):
        assert fm.calculate_bleu([ref], hyp, threshold=0.8) == expected[i]

def test_bleu_match_commons_tokenizer():
    expected = [False, True, True]
    fm = FuzzyMatch(tokenizer=commons_tokenizer)
    for i, (ref, hyp) in enumerate(zip(references, inputs)):
        assert fm.calculate_bleu([ref], hyp, threshold=0.8) == expected[i]