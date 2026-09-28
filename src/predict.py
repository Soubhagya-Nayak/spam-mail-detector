"""Load the trained model and classify messages.

Usage:
    python -m src.predict "Congratulations! You won a free prize, click now"
    python -m src.predict            # interactive mode
"""
import sys
from pathlib import Path

import joblib

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "spam_model.joblib"


class SpamDetector:
    def __init__(self, model_path=MODEL_PATH, threshold: float = 0.5):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained model at {model_path}. Run: python -m src.train")
        self.model = joblib.load(model_path)
        self.threshold = threshold

    def predict(self, text: str) -> dict:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        prob = float(self.model.predict_proba([text])[0][1])
        is_spam = prob >= self.threshold
        return {
            "label": "spam" if is_spam else "ham",
            "is_spam": is_spam,
            "spam_probability": round(prob, 4),
        }


def _print(result: dict):
    print(f"{result['label'].upper():<5} (spam probability: {result['spam_probability']:.2%})")


def main():
    detector = SpamDetector()
    if len(sys.argv) > 1:
        _print(detector.predict(" ".join(sys.argv[1:])))
        return
    print("Type a message (empty line to quit):")
    while True:
        line = input("> ").strip()
        if not line:
            break
        _print(detector.predict(line))


if __name__ == "__main__":
    main()
