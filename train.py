import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FAKE_PATH = os.path.join(BASE_DIR, "Fake.csv")
REAL_PATH = os.path.join(BASE_DIR, "True.csv")
MODEL_PATH = os.path.join(BASE_DIR, "fake_news_model.joblib")


def load_dataset():
    """Load and combine real and fake news datasets."""

    if not os.path.exists(FAKE_PATH) or not os.path.exists(REAL_PATH):
        raise FileNotFoundError(
            "Dataset files not found!\n"
            "Place Fake.csv and True.csv in the project directory."
        )

    fake_df = pd.read_csv(FAKE_PATH)
    real_df = pd.read_csv(REAL_PATH)

    # Assign labels to each category.
    fake_df["label"] = 0
    real_df["label"] = 1

    # Combine title and article text.
    for df in (fake_df, real_df):
        if "text" not in df.columns:
            raise ValueError("Dataset must contain a 'text' column.")

        if "title" in df.columns:
            df["text"] = (
                df["title"].fillna("").astype(str)
                + " "
                + df["text"].fillna("").astype(str)
            )
        else:
            df["text"] = df["text"].fillna("").astype(str)

    data = pd.concat(
        [
            fake_df[["text", "label"]],
            real_df[["text", "label"]],
        ],
        ignore_index=True,
    )

    # Remove empty articles and duplicates.
    data["text"] = data["text"].str.strip()
    data = data[data["text"] != ""]
    data = data.drop_duplicates(subset="text")

    # Shuffle the dataset.
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    return data


def train_model():
    """Train and evaluate the Fake News Detection model."""

    data = load_dataset()

    X = data["text"]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                stop_words="english",
                max_features=50000,
                ngram_range=(1, 2),
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42,
            ),
        ),
    ])

    print("\nTraining the model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\n========== MODEL EVALUATION ==========")
    print(f"Total articles: {len(data)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Accuracy: {accuracy:.2%}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["FAKE", "REAL"],
            zero_division=0,
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    joblib.dump(model, MODEL_PATH)

    print(f"\nModel saved successfully: {MODEL_PATH}")

    return model


if __name__ == "__main__":
    train_model()