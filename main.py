import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fake_news_model.joblib")


def main():
    print("=" * 50)
    print("       FAKE NEWS DETECTION SYSTEM")
    print("=" * 50)

    if not os.path.exists(MODEL_PATH):
        print("\nTrained model not found.")
        print("Run this command first: python train.py")
        return

    model = joblib.load(MODEL_PATH)

    print("\nModel loaded successfully!")
    print("Enter a news article or headline to analyze.")
    print("Type 'exit' to close the application.")

    while True:
        print("\n" + "-" * 50)

        news = input("Enter news text: ").strip()

        if news.lower() == "exit":
            print("Exiting the application.")
            break

        if not news:
            print("Please enter some news text.")
            continue

        prediction = model.predict([news])[0]
        probabilities = model.predict_proba([news])[0]

        class_index = list(model.classes_).index(prediction)
        score = probabilities[class_index] * 100

        print("\n========== PREDICTION ==========")

        if prediction == 0:
            print("Predicted Category: FAKE")
        else:
            print("Predicted Category: REAL")

        print(f"Model confidence score: {score:.2f}%")

        print(
            "\nDisclaimer: This is a Machine Learning prediction, "
            "not a verified fact-check. The model can be wrong."
        )


if __name__ == "__main__":
    main()