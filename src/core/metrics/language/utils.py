import string
import re

def clean_text(text: str) -> str:
    # Remove punctuation and extra whitespace, lowercasing
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    return text.strip()