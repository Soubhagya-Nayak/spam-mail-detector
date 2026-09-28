"""Train, compare and save the spam classifier.

Usage:
    python -m src.train                                   # SMS Spam Collection
    python -m src.train --data emails.csv --text-col text --label-col label
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .preprocess import clean_text

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "spam_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
CM_PATH = ROOT / "models" / "confusion_matrix.png"
RANDOM_STATE = 42


def load_dataset(path, text_col=None, label_col=None) -> pd.DataFrame:
    """Load a labelled dataset into a DataFrame with `text` and `label` (1=spam)."""
    path = Path(path)
    if text_col is None:  # default: headerless "label<TAB>text" file
        df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"])
    else:
        df = pd.read_csv(path, encoding="latin-1")
        df = df.rename(columns={label_col: "label", text_col: "text"})[["label", "text"]]
    df = df.dropna().drop_duplicates()  # duplicates would leak between train/test
    df["label"] = df["label"].astype(str).str.lower().map(
        {"spam": 1, "ham": 0, "1": 1, "0": 0})
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    return df.reset_index(drop=True)


def build_pipeline(classifier) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2),
                                  min_df=2, sublinear_tf=True)),
        ("clf", classifier),
    ])


def candidate_models() -> dict:
    return {
        "naive_bayes": MultinomialNB(alpha=0.1),
        "logistic_regression": LogisticRegression(
            max_iter=1000, C=10, class_weight="balanced"),
        # LinearSVC has no predict_proba, so calibrate it to get probabilities
        "linear_svm": CalibratedClassifierCV(
            LinearSVC(class_weight="balanced"), cv=5),
    }


def save_confusion_matrix(cm, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Ham", "Spam"])
    ax.set_yticks([0, 1], ["Ham", "Spam"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (test set)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i][j]), ha="center", va="center",
                    color="white" if cm[i][j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train the spam classifier")
    parser.add_argument("--data", default=str(ROOT / "data" / "sms.tsv"))
    parser.add_argument("--text-col")
    parser.add_argument("--label-col")
    args = parser.parse_args()

    df = load_dataset(args.data, args.text_col, args.label_col)
    print(f"Loaded {len(df)} unique messages "
          f"({df.label.sum()} spam / {(df.label == 0).sum()} ham)")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, stratify=df["label"],
        random_state=RANDOM_STATE)

    # Pick the model with cross-validation on the TRAIN split only,
    # so the test set stays untouched for the final report.
    cv_scores = {}
    for name, clf in candidate_models().items():
        scores = cross_val_score(build_pipeline(clf), X_train, y_train,
                                 cv=5, scoring="f1")
        cv_scores[name] = float(scores.mean())
        print(f"  {name:<20} CV F1 = {scores.mean():.4f}")

    best_name = max(cv_scores, key=cv_scores.get)
    print(f"Best model: {best_name}")

    model = build_pipeline(candidate_models()[best_name]).fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "best_model": best_name,
        "cv_f1": cv_scores,
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred),
        "test_recall": recall_score(y_test, y_pred),
        "test_f1": f1_score(y_test, y_pred),
        "confusion_matrix": cm.tolist(),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    print("\n" + classification_report(y_test, y_pred, target_names=["ham", "spam"]))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    save_confusion_matrix(cm, CM_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
