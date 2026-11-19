import argparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

TRAINING_TEXTS = [
    "hello how are you today", "the weather is very nice", "thank you for your help",
    "hola como estas hoy", "el clima es muy bueno", "gracias por tu ayuda",
    "bonjour comment allez vous", "le temps est tres beau", "merci pour votre aide",
]
TRAINING_LANGUAGES = ["english"] * 3 + ["spanish"] * 3 + ["french"] * 3


def build_model() -> Pipeline:
    model = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 5))),
        ("classifier", MultinomialNB()),
    ])
    model.fit(TRAINING_TEXTS, TRAINING_LANGUAGES)
    return model


def detect_language(text: str) -> str:
    if not text.strip():
        raise ValueError("Text cannot be empty")
    return str(build_model().predict([text])[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect the language of a short sentence.")
    parser.add_argument("text", nargs="+", help="Sentence to classify")
    args = parser.parse_args()
    print(f"Detected language: {detect_language(' '.join(args.text))}")


if __name__ == "__main__":
    main()
