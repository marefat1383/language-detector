# Language Detector

A small educational AI project that detects whether a short sentence is English, Spanish, or French using character n-gram TF-IDF features and Naive Bayes.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Run

```bash
python language_detector.py "Hello, how are you today?"
```

## Test

```bash
python -m unittest test_language_detector.py
```

The training examples are intentionally small and embedded in the source for a self-contained demonstration.
