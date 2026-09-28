import app as app_module


class FakeDetector:
    def predict(self, text):
        return {"label": "spam", "is_spam": True, "spam_probability": 0.99}


def make_client(monkeypatch):
    monkeypatch.setattr(app_module, "_detector", FakeDetector())
    return app_module.app.test_client()


def test_health(monkeypatch):
    assert make_client(monkeypatch).get("/health").json == {"status": "ok"}


def test_predict_ok(monkeypatch):
    res = make_client(monkeypatch).post("/api/predict", json={"text": "free money"})
    assert res.status_code == 200
    assert res.json["label"] == "spam"


def test_predict_rejects_empty(monkeypatch):
    res = make_client(monkeypatch).post("/api/predict", json={"text": "  "})
    assert res.status_code == 400


def test_predict_rejects_too_long(monkeypatch):
    res = make_client(monkeypatch).post("/api/predict", json={"text": "a" * 10_001})
    assert res.status_code == 413
