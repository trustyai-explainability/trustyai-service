import pytest
from src.core.metrics.language.error_rates.word_error_rate import WordErrorRate


references = [
    "This is the test reference, to which I will compare alignment against.",
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Curabitur condimentum velit id velit posuere dictum. Fusce euismod tortor massa, nec euismod sapien laoreet non. Donec vulputate mi velit, eu ultricies nibh iaculis vel. Aenean posuere urna nec sapien consectetur, vitae porttitor sapien finibus. Duis nec libero convallis lectus pharetra blandit ut ac odio. Vivamus nec dui quis sem convallis pulvinar. Maecenas sodales sollicitudin leo a faucibus.",
    "The quick red fox jumped over the lazy brown dog",
    "i love cold pizza"
]

inputs = [
    "I'm a hypothesis reference, from which the aligner  will compare against.",
    "Lorem ipsum sit amet, consectetur adipiscing elit. Curabitur condimentum velit id velit posuere dictum. Fusce blandit euismod tortor massa, nec euismod sapien blandit laoreet non. Donec vulputate mi velit, eu ultricies nibh iaculis vel. Aenean posuere urna nec sapien consectetur, vitae porttitor sapien finibus. Duis nec libero convallis lectus pharetra blandit ut ac odio. Vivamus nec dui quis sem convallis pulvinar. Maecenas sodales sollicitudin leo a faucibus.",
    "dog brown lazy the over jumped fox red quick The",
    "i love pizza"
]


ground_truth_commons = [8 / 12., 3 / 66., 1.0, 0.25]
ground_truth_nlp = [9 / 14., 3 / 78., 1.0, 0.25]
ground_truth_whitespace = [8 / 14., 3 / 78., 10 / 10., 0.25]

TOLERANCE = 0.1



def commons_tokenizer(text: str):
    """Simulates Apache StringTokenizer (split on whitespace, drop empty tokens)."""
    return [token for token in text.split() if token.strip()]

def whitespace_tokenizer(text: str):
    """Simulates Java-style default whitespace split."""
    return text.split()

def simulated_open_nlp_tokenizer(text: str):
    """Rough simulation of OpenNLP's SimpleTokenizer with basic punctuation split."""
    import re
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)



@pytest.mark.parametrize("ref, hyp, expected", zip(references, inputs, ground_truth_commons))
def test_commons_tokenizer(ref, hyp, expected):
    wer = WordErrorRate(tokenizer=commons_tokenizer)
    result = wer.calculate(ref, hyp)
    assert pytest.approx(result.value, abs=TOLERANCE) == expected

@pytest.mark.parametrize("ref, hyp, expected", zip(references, inputs, ground_truth_whitespace))
def test_whitespace_tokenizer_default(ref, hyp, expected):
    wer = WordErrorRate(tokenizer=whitespace_tokenizer)
    result = wer.calculate(ref, hyp)
    assert pytest.approx(result.value, abs=TOLERANCE) == expected

@pytest.mark.parametrize("ref, hyp, expected", zip(references, inputs, ground_truth_nlp))
def test_simulated_open_nlp_tokenizer(ref, hyp, expected):
    wer = WordErrorRate(tokenizer=simulated_open_nlp_tokenizer)
    result = wer.calculate(ref, hyp)
    assert pytest.approx(result.value, abs=TOLERANCE) == expected