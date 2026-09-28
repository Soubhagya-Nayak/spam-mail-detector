"""Text cleaning shared by training and inference."""
import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_URL = re.compile(r"(https?://\S+|www\.\S+)", re.I)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_MONEY = re.compile(r"[$£€₹]\s?\d[\d,.]*")
_PHONE = re.compile(r"\b\d{7,}\b")
_NUMBER = re.compile(r"\b\d+\b")
_NON_ALPHA = re.compile(r"[^a-z\s]")


def clean_text(text: str) -> str:
    """Lowercase, replace URLs/emails/money/phones/numbers with tokens, drop stop words.

    Replacing them with placeholder tokens (instead of deleting them) keeps a
    strong spam signal: e.g. messages full of links, prices and phone numbers.
    """
    text = str(text).lower()
    text = _URL.sub(" urltoken ", text)
    text = _EMAIL.sub(" emailtoken ", text)
    text = _MONEY.sub(" moneytoken ", text)
    text = _PHONE.sub(" phonetoken ", text)
    text = _NUMBER.sub(" numtoken ", text)
    text = _NON_ALPHA.sub(" ", text)
    tokens = [t for t in text.split() if len(t) > 1 and t not in ENGLISH_STOP_WORDS]
    return " ".join(tokens)
