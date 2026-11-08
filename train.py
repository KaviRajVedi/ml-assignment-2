"""Train offline, persist pipelines, and regenerate measured report sections."""
import hashlib
import json
import platform
from importlib.metadata import version
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from config import ROOT, SEED, MODEL_FILES, LABELS, CLASS_NAMES
from evaluation import validate_dataset, evaluate_classifier


def build_models():
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED)),
        "Decision Tree": make_pipeline(DecisionTreeClassifier(max_depth=5, min_samples_leaf=3, random_state=SEED)),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "Naive Bayes": make_pipeline(GaussianNB()),
        "Support Vector Machine": make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf", probability=True, random_state=SEED)),
    }


def markdown_table(frame):
    header = "| " + " | ".join(frame.columns) + " |"
    divider = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
    lines = [header, divider]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(lines)


def main():
    features = list(load_breast_cancer().feature_names)
    paths = [ROOT / "data/train_data.csv", ROOT / "test_data.csv"]
    if not all(path.exists() for path in paths):
        raise SystemExit("Missing data. Run python prepare_data.py first.")
    train_frame, test_frame = [pd.read_csv(path) for path in paths]
    x_train, y_train = validate_dataset(train_frame, features)
    x_test, y_test = validate_dataset(test_frame, features)
    if y_train.nunique() != 2 or y_test.nunique() != 2:
        raise ValueError("The training and held-out splits must both contain both classes.")
    if not x_train.merge(x_test).empty:
        raise ValueError("Train/test feature overlap detected; recreate the held-out split.")
    (ROOT / "model").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    rows, observations, details, train_scores = [], [], {}, []
    explanations = {
        "Logistic Regression": "A scaled linear baseline; nonlinear relationships are not explicitly represented.",
        "Decision Tree": "Depth is capped at 5 and leaves need 3 samples to limit complexity; probabilities are leaf proportions.",
        "KNN": "Uses five neighbours in standardized feature space; all features contribute to distance.",
        "Naive Bayes": "Uses raw measurements and class-conditional Gaussian distributions with an independence assumption.",
        "Support Vector Machine": "Uses standardized features and an RBF kernel; probabilities use training-only internal calibration.",
    }
    for name, model in build_models().items():
        model.fit(x_train, y_train)
        metrics = evaluate_classifier(model, x_test, y_test)
        training = evaluate_classifier(model, x_train, y_train)
        predictions = model.predict(x_test)
        matrix = confusion_matrix(y_test, predictions, labels=LABELS)
        details[name] = {
            "confusion_matrix": matrix.tolist(), "label_order": CLASS_NAMES,
            "classification_report": classification_report(y_test, predictions, labels=LABELS,
                target_names=CLASS_NAMES, output_dict=True, zero_division=0),
        }
        joblib.dump(model, ROOT / "model" / MODEL_FILES[name], compress=3)
        rows.append({"ML Model Name": name, **metrics})
        train_scores.append({"ML Model Name": name, **training})
        observations.append({"ML Model Name": name, "Observation about model performance":
            f"Test F1 {metrics['F1']:.4f}, malignant recall {metrics['Recall']:.4f}, "
            f"AUC {metrics['AUC']:.4f}. Missed {matrix[0, 1]} of {sum(matrix[0])} malignant cases "
            f"and flagged {matrix[1, 0]} benign cases as malignant. Train/test accuracy "
            f"{training['Accuracy']:.4f}/{metrics['Accuracy']:.4f}. " + explanations[name]})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(ROOT / "results/metrics.csv", index=False)
    pd.DataFrame(train_scores).to_csv(ROOT / "results/train_metrics.csv", index=False)
    (ROOT / "results/classification_reports.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    winner = comparison.sort_values(["F1", "Recall", "AUC"], ascending=False).iloc[0]
    observations.append({"ML Model Name": "Overall winner for this split",
        "Observation about model performance": f"{winner['ML Model Name']} by malignant-class F1 ({winner['F1']:.4f}); "
        "ties are broken by recall then AUC. This is a descriptive comparison on one small held-out split, "
        "not evidence of clinical suitability or a universally best model."})
    report = markdown_table(comparison) + "\n\n" + markdown_table(pd.DataFrame(observations))
    (ROOT / "results/comparison.md").write_text(report + "\n", encoding="utf-8")
    readme = ROOT / "README.md"
    content = readme.read_text(encoding="utf-8")
    start, end = "<!-- RESULTS_START -->", "<!-- RESULTS_END -->"
    if start in content and end in content:
        before = content.split(start)[0]
        after = content.split(end, 1)[1]
        readme.write_text(before + start + "\n" + report + "\n" + end + after, encoding="utf-8")
    packages = {name: version(name) for name in ["scikit-learn", "numpy", "pandas", "scipy", "joblib", "streamlit"]}
    metadata = {"python": platform.python_version(), "packages": packages, "seed": SEED,
        "positive_class": 0, "features": features, "train_rows": len(x_train), "test_rows": len(x_test),
        "train_class_counts": y_train.value_counts().sort_index().to_dict(),
        "test_class_counts": y_test.value_counts().sort_index().to_dict(),
        "data_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (ROOT / "model/metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("Saved five pipelines, evaluation reports and refreshed README results.")

if __name__ == "__main__":
    main()
