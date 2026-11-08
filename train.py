"""Training exercise. The web app must never call training on uploaded data."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

def main():
    # 1. Read data/train_data.csv and test_data.csv; separate target from X.
    # 2. Construct LogisticRegression, DecisionTreeClassifier,
    #    KNeighborsClassifier, GaussianNB, and SVC.
    # 3. Use StandardScaler in a Pipeline for LR, KNN and SVM.
    #    SVC(probability=True) exposes predict_proba; set a random_state.
    # 4. Fit all five on the same training rows only. Use training-only CV
    #    if selecting hyperparameters. Record the chosen settings.
    # 5. Call evaluation.evaluate_classifier on the held-out rows.
    # 6. Save each complete pipeline to model/ with joblib.dump.
    # 7. Save the comparison table to results/metrics.csv and inspect
    #    per-class classification reports and confusion matrices.
    raise NotImplementedError("Complete the training steps, then run on BITS Virtual Lab.")

if __name__ == "__main__":
    main()
