"""Shared dataset and artifact conventions."""
from pathlib import Path
ROOT = Path(__file__).resolve().parent
TARGET = "target"
POSITIVE_CLASS = 0
LABELS = [0, 1]
CLASS_NAMES = ["Malignant", "Benign"]
SEED = 17
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}
