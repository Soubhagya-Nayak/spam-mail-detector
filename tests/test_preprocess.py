from src.preprocess import clean_text


def test_replaces_urls_and_numbers():
    out = clean_text("Win $500 now at http://free-prize.com call 09012345678")
    assert "urltoken" in out
    assert "moneytoken" in out
    assert "phonetoken" in out
    assert "http" not in out


def test_removes_stopwords_and_punctuation():
    assert clean_text("The cat, and the hat!!!") == "cat hat"


def test_handles_non_string():
    assert clean_text(12345) == "numtoken"
