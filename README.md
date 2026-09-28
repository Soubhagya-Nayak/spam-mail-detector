# Spam Mail Detector

![CI](https://github.com/<your-username>/spam-mail-detector/actions/workflows/ci.yml/badge.svg)

A machine-learning project that classifies messages as **spam** or **ham** (not spam). It includes a training pipeline that compares three models, a command-line predictor, a Flask REST API, a small web UI, unit tests, and GitHub Actions CI.

## Results

Trained on the [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) (5,169 unique messages after removing duplicates), 80/20 stratified split:

| Metric (test set) | Score |
|---|---|
| Accuracy | 98.5% |
| Precision (spam) | 93.9% |
| Recall (spam) | 93.9% |
| F1 (spam) | 93.9% |

Model selection uses 5-fold cross-validation on the training split only, so the test set is never used to pick the model.

| Model | CV F1 |
|---|---|
| Naive Bayes | 0.940 |
| **Logistic Regression** (selected) | **0.944** |
| Linear SVM (calibrated) | 0.941 |

Re-run `python -m src.train` to reproduce these numbers; they are also saved in `models/metrics.json`.

## How it works

1. **Preprocessing** (`src/preprocess.py`): lowercase, replace URLs / emails / money / phone numbers / digits with placeholder tokens, strip punctuation, remove stop words.
2. **Features**: TF-IDF with unigrams + bigrams, sublinear term frequency.
3. **Models**: Multinomial Naive Bayes, Logistic Regression, Linear SVM (probability-calibrated).
4. **Serving**: `SpamDetector` class → CLI, Flask JSON API, web page.

## Project structure

```
spam-mail-detector/
├── app.py                  # Flask web app + REST API
├── src/
│   ├── preprocess.py       # text cleaning
│   ├── train.py            # training, model comparison, evaluation
│   └── predict.py          # inference class + CLI
├── templates/index.html    # web UI
├── data/sms.tsv            # dataset (SMS Spam Collection)
├── models/                 # trained model, metrics, confusion matrix
├── tests/                  # pytest suite
├── .github/workflows/ci.yml
└── requirements.txt
```

## Getting started

```bash
git clone https://github.com/<your-username>/spam-mail-detector.git
cd spam-mail-detector
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.train              # trains and saves models/spam_model.joblib
```

### Command line

```bash
python -m src.predict "WINNER!! Claim your free prize now at http://bit.ly/x"
```

### Web app

```bash
python app.py                    # open http://127.0.0.1:5000
```

### REST API

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "Free entry! Text WIN to 80086 to claim your cash prize"}'
```

```json
{"is_spam": true, "label": "spam", "spam_probability": 0.9812}
```

### Tests

```bash
pytest -q
```

## Train on your own email dataset

Any CSV with a text column and a label column (`spam`/`ham` or `1`/`0`) works:

```bash
python -m src.train --data emails.csv --text-col text --label-col label
```

## Limitations

- The bundled dataset is short SMS text from the 2000s. It works well on that style of message, but may miss modern phishing emails or long emails. Retraining on an email corpus (see above) is recommended for real email filtering.
- Classical bag-of-words models can be evaded by obfuscated wording; transformer-based models would be a natural next step.

## Future improvements

- Retrain on a large email dataset (e.g. Enron-Spam) and compare results
- Add a DistilBERT baseline
- Gmail integration to scan an inbox
- Docker image and cloud deployment

## Acknowledgements

Dataset: Almeida, T.A., Gómez Hidalgo, J.M., Yamakami, A. *Contributions to the Study of SMS Spam Filtering* (2011), via the UCI Machine Learning Repository.

## License

MIT
