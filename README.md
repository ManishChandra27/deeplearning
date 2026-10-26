# Fake News Detection using Machine Learning

## 1. Project Overview

Fake News Detection is a Machine Learning project that classifies
news articles as FAKE or REAL based on their textual content.

The project uses Natural Language Processing (NLP), TF-IDF
feature extraction, and Logistic Regression for classification.

## 2. Objectives

- Apply Machine Learning to a real-world problem.
- Understand text preprocessing and feature extraction.
- Implement a supervised classification algorithm.
- Evaluate the model using standard classification metrics.
- Build a simple application for predicting news categories.

## 3. Technologies Used

- Python
- Pandas
- Scikit-learn
- Joblib
- Natural Language Processing (NLP)

## 4. Machine Learning Algorithms

### TF-IDF

Term Frequency-Inverse Document Frequency converts text into
numerical features.

It gives greater importance to words that are informative
within a document and less importance to words common across
many documents.

### Logistic Regression

Logistic Regression is a supervised Machine Learning algorithm
used for classification.

It learns relationships between text features and the labels
FAKE and REAL.

### Pipeline

The Scikit-learn Pipeline combines TF-IDF vectorization and
Logistic Regression into a single model.

This ensures that the same feature extraction process is used
during training and prediction.

## 5. Project Structure

    FakeNewsDetection/
    ├── train.py
    ├── main.py
    ├── README.md
    ├── Fake.csv
    ├── True.csv
    └── fake_news_model.joblib

The CSV files are input datasets. The model file is generated
automatically after training.

## 6. Dataset

Download the Fake and Real News Dataset from Kaggle:

https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Place Fake.csv and True.csv in the project directory.

Fake.csv contains examples labeled FAKE.
True.csv contains examples labeled REAL.

The dataset labels are treated as ground truth for training.
They do not independently establish that every article is
factually false or true.

## 7. Installation

Install Python and the required libraries.

    pip install pandas scikit-learn joblib

## 8. Execution

### Step 1: Train the model

    python train.py

This command loads the dataset, trains the model, evaluates
its performance, and saves fake_news_model.joblib.

### Step 2: Run the application

    python main.py

Enter a news headline or article when prompted.

The application displays the predicted category and model
confidence score.

Type exit to close the application.

## 9. Evaluation Metrics

### Accuracy

The proportion of predictions that are correct.

### Precision

The proportion of predicted positive examples that are
actually positive according to the test labels.

### Recall

The proportion of actual positive examples correctly
identified by the model.

### F1-Score

The harmonic mean of precision and recall.

### Confusion Matrix

A table showing correct and incorrect predictions for
each class.

Actual evaluation results are displayed when train.py runs.
No performance score is assumed in advance.

## 10. Limitations

- The model learns patterns from its training dataset.
- New topics and writing styles may reduce performance.
- The dataset may contain biases or misleading labels.
- Text classification cannot independently verify factual claims.
- The confidence score is not proof of truth or falsehood.

## 11. Future Improvements

- Use a larger and more diverse dataset.
- Test the model on articles from unseen sources.
- Add explainability to show influential text features.
- Integrate trusted fact-checking sources.
- Develop a web interface for easier interaction.

## 12. Conclusion

This project demonstrates how supervised Machine Learning
and Natural Language Processing can be applied to text
classification.

It provides practical experience with feature extraction,
model training, evaluation, and prediction while highlighting
the limitations of automated misinformation detection.