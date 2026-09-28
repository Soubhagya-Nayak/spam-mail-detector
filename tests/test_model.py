import pytest

from src.train import ROOT, build_pipeline, candidate_models, load_dataset

DATA = ROOT / "data" / "sms.tsv"


@pytest.fixture(scope="module")
def model():
    df = load_dataset(DATA)
    return build_pipeline(candidate_models()["logistic_regression"]).fit(
        df["text"], df["label"])


def test_detects_obvious_spam(model):
    text = "WINNER!! You have won a free prize. Call 09061701461 now to claim your cash!"
    assert model.predict([text])[0] == 1


def test_accepts_normal_message(model):
    assert model.predict(["Hey, are we still meeting for lunch tomorrow?"])[0] == 0


def test_probability_range(model):
    p = model.predict_proba(["free entry to win cash, text now"])[0][1]
    assert 0.0 <= p <= 1.0
