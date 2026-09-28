"""Flask web app + JSON API for the spam detector."""
from flask import Flask, jsonify, render_template, request

from src.predict import SpamDetector

app = Flask(__name__)
MAX_CHARS = 10_000
_detector = None


def get_detector() -> SpamDetector:
    global _detector
    if _detector is None:
        _detector = SpamDetector()
    return _detector


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.post("/api/predict")
def api_predict():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    if not isinstance(text, str) or not text.strip():
        return jsonify(error="Field 'text' is required and must be non-empty."), 400
    if len(text) > MAX_CHARS:
        return jsonify(error=f"Text too long (max {MAX_CHARS} characters)."), 413
    return jsonify(get_detector().predict(text))


if __name__ == "__main__":
    app.run(debug=True)
