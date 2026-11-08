"""Evaluate trusted saved pipelines. Uploaded data is never used for training."""
import csv
import io
import json
from importlib.metadata import version
import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix
from config import ROOT, MODEL_FILES, LABELS, CLASS_NAMES
from evaluation import validate_dataset, evaluate_classifier, positive_probability

st.set_page_config(page_title="Tumour Classification Lab", page_icon=":microscope:", layout="wide")
st.title("Tumour Classification Lab")
st.caption("Machine Learning Assignment 2 | Wisconsin Diagnostic dataset")
st.info("Educational model comparison. Positive class: malignant (0); benign: 1. Not for clinical use.")
st.markdown("Explore the dataset, compare five classifiers, and understand their predictions on held-out samples.")
st.sidebar.title("Evaluation controls")
choice = st.selectbox("Classifier", list(MODEL_FILES), key="classifier", help="Choose the model shown in Model evaluation. The comparison includes all five models.")
upload = st.sidebar.file_uploader("Upload test CSV (features + target)", type=["csv"])
use_sample = st.sidebar.checkbox("Use bundled test data when no CSV is uploaded", value=True)
bundled = ROOT / "test_data.csv"
if bundled.exists():
    st.sidebar.download_button("Download test CSV", bundled.read_bytes(), "test_data.csv", "text/csv")
st.sidebar.caption("Uploads take priority. All models are compared on the same selected rows.")
st.sidebar.divider()
st.sidebar.markdown("**Project links**")
st.sidebar.link_button("GitHub repository", "https://github.com/KaviRajVedi/ml-assignment-2")
st.sidebar.link_button("UCI dataset", "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic")
with st.sidebar.expander("CSV requirements"):
    st.write("Include all 30 numeric features and target: 0 = malignant, 1 = benign. Column order may vary; extra columns are ignored.")
    st.write("No missing or infinite values. Maximum 10 MB and 10,000 rows. Upload held-out data only.")

metadata_path = ROOT / "model/metadata.json"
missing = [filename for filename in MODEL_FILES.values() if not (ROOT / "model" / filename).exists()]
if missing or not metadata_path.exists():
    st.warning("Models are not ready. Run python prepare_data.py, then python train.py in your terminal.")
    st.stop()
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
mismatches = [f"{name}: trained with {expected}, installed {version(name)}"
              for name, expected in metadata["packages"].items()
              if name in {"scikit-learn", "numpy", "scipy", "joblib", "pandas"} and version(name) != expected]
if mismatches:
    st.error("Model environment mismatch. Install the pinned requirements or retrain locally with python train.py.")
    st.code("\n".join(mismatches))
    st.stop()

@st.cache_resource
def load_models(signature):
    # Only repository-controlled artifacts are deserialized; uploads are CSV only.
    return {name: joblib.load(ROOT / "model" / filename) for name, filename in MODEL_FILES.items()}

try:
    signature = tuple((filename, (ROOT / "model" / filename).stat().st_mtime_ns) for filename in MODEL_FILES.values())
    models = load_models(signature)
except Exception as exc:
    st.error(f"Unable to load saved models. Run python train.py again. Details: {exc}")
    st.stop()

if upload is not None:
    raw = upload.getvalue()
elif use_sample and bundled.exists():
    raw = bundled.read_bytes()
else:
    st.info("Upload a test CSV or select the bundled test data.")
    st.stop()
try:
    if len(raw) > 10 * 1024 * 1024:
        raise ValueError("Upload a test CSV smaller than 10 MB.")
    text = raw.decode("utf-8-sig")
    header = next(csv.reader(io.StringIO(text)), [])
    if len(header) != len(set(header)):
        raise ValueError("Duplicate column names are not allowed.")
    frame = pd.read_csv(io.StringIO(text))
    if len(frame) > 10000:
        raise ValueError("Upload at most 10,000 test rows.")
    features, target = validate_dataset(frame, metadata["features"])
except (ValueError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
    st.error(f"Invalid test CSV: {exc}")
    st.stop()

comparison = pd.DataFrame([{"ML Model Name": name, **evaluate_classifier(model, features, target)}
                           for name, model in models.items()]).set_index("ML Model Name")
st.caption(f"Evaluating {len(features)} rows and {len(features.columns)} features; models are not retrained.")
if target.nunique() < 2:
    st.warning("Only one class is present: AUC and MCC are shown as N/A. Other metrics describe this subset only.")
st.caption("Precision, recall and F1 refer to malignant cases. Undefined precision/recall/F1 are reported as 0.")
overview_tab, evaluation_tab, comparison_tab, data_tab, guide_tab = st.tabs(
    ["Project overview", "Model evaluation", "Compare all five", "Explore test data", "Methods and README"])
with overview_tab:
    st.subheader("From cell measurements to a classification")
    st.write("The task is to distinguish malignant and benign breast tumour samples using measurements of cell nuclei from digitized fine-needle aspirate images. Five supervised classifiers are evaluated on the same held-out partition.")
    for column, title, value, description in zip(st.columns(4),
            ["Full dataset", "Input features", "Training partition", "Held-out test partition"],
            ["569 samples", "30 measurements", "455 samples", "114 samples"],
            ["212 malignant and 357 benign", "10 properties, 3 summaries each", "80% of the data", "20% reserved for evaluation"]):
        with column.container(border=True):
            st.caption(title)
            st.subheader(value)
            st.caption(description)
    st.success("Dataset requirements met: 569 total instances exceed the minimum of 500, and 30 features exceed the minimum of 12. The 114 rows in the app are the test partition, not the full dataset.")
    left, right = st.columns(2)
    with left:
        st.markdown("#### How to use this app")
        st.markdown("1. Use the bundled test data or upload a labeled test CSV.\n2. Pick a classifier using the dropdown above.\n3. Open **Model evaluation** for scores and mistakes.\n4. Use **Compare all five** to compare the same input rows.\n5. Inspect every sample in **Explore test data**.")
    with right:
        st.markdown("#### What the labels mean")
        st.write("**0 - Malignant:** the positive class for precision, recall, F1 and AUC.")
        st.write("**1 - Benign:** the negative class.")
        st.write("A false negative is a malignant sample predicted benign. A false positive is a benign sample predicted malignant.")
    st.markdown("#### Current evaluation")
    st.write(f"Source: {'uploaded CSV' if upload is not None else 'bundled held-out test CSV'}. "
             f"{len(target)} rows: {(target == 0).sum()} malignant and {(target == 1).sum()} benign.")
    st.caption("The dataset and split counts above describe the original experiment. The current evaluation counts change when you upload another CSV. Uploaded data is evaluated, never used for fitting.")
with evaluation_tab:
    st.subheader(choice)
    st.caption("Scores below are calculated from the current CSV, with malignant as the positive class.")
    for column, (metric, value) in zip(st.columns(6), comparison.loc[choice].items()):
        column.metric(metric, "N/A" if pd.isna(value) else f"{value:.4f}")
    predictions = models[choice].predict(features)
    matrix = confusion_matrix(target, predictions, labels=LABELS)
    st.subheader("Confusion matrix")
    st.caption("Rows = actual class; columns = predicted class. Malignant predicted as Benign is a missed malignant case.")
    st.dataframe(pd.DataFrame(matrix, index=["Actual " + n for n in CLASS_NAMES],
                             columns=["Predicted " + n for n in CLASS_NAMES]), width="stretch")
    st.write(f"**{matrix[0, 0]}** malignant cases correctly identified; **{matrix[0, 1]}** malignant cases missed. "
             f"**{matrix[1, 1]}** benign cases correctly identified; **{matrix[1, 0]}** benign cases flagged as malignant.")
    st.subheader("Classification report")
    report = classification_report(target, predictions, labels=LABELS,
                                   target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T, width="stretch")
    output = features.copy()
    output["target"] = target
    output["prediction"] = predictions
    output["malignant_probability"] = positive_probability(models[choice], features)
    st.download_button("Download predictions", output.to_csv(index=False), "predictions.csv", "text/csv")
    with st.expander("Inspect predictions and errors", expanded=False):
        prediction_view = output.copy()
        prediction_view.insert(0, "CSV row", range(1, len(output) + 1))
        prediction_view.insert(1, "Actual class", target.map({0: "Malignant", 1: "Benign"}))
        prediction_view.insert(2, "Predicted class", pd.Series(predictions, index=target.index).map({0: "Malignant", 1: "Benign"}))
        only_errors = st.checkbox("Show misclassified rows only", key="only_errors")
        if only_errors:
            prediction_view = prediction_view.loc[target != predictions]
        st.caption(f"Showing {len(prediction_view)} rows. CSV row counts data rows, excluding the header. Probabilities are model estimates, not clinical risk assessments.")
        st.dataframe(prediction_view, width="stretch", hide_index=True)
with comparison_tab:
    st.subheader("Same data, five classifiers")
    st.caption("Every score below uses the current evaluation CSV. Higher is better for all six metrics; MCC can range from -1 to 1.")
    # Format a display copy without pandas Styler's optional Jinja2 dependency.
    # Keep comparison numeric for charts and CSV downloads.
    display_comparison = comparison.map(lambda value: "N/A" if pd.isna(value) else f"{value:.4f}")
    st.dataframe(display_comparison, width="stretch")
    metric = st.selectbox("Metric to compare", list(comparison.columns), index=4)
    if comparison[metric].notna().any():
        st.bar_chart(comparison[[metric]])
    else:
        st.info(f"{metric} requires both classes in the uploaded data.")
    st.download_button("Download comparison", comparison.to_csv(), "comparison.csv", "text/csv")
    if target.nunique() == 2:
        ranked = comparison.sort_values(["F1", "Recall", "AUC"], ascending=False)
        best = ranked.iloc[0]
        tied = ranked.index[(ranked[["F1", "Recall", "AUC"]] == best[["F1", "Recall", "AUC"]]).all(axis=1)].tolist()
        st.success(f"Best on this CSV by malignant-class F1: {', '.join(tied)} (F1 {best['F1']:.4f}). Recall and AUC break F1 ties.")
    else:
        st.info("No overall winner is declared for a single-class subset. Upload both classes for a meaningful two-class comparison.")
    st.caption("This is a descriptive comparison of the selected rows, not proof of a universally best model. Small test sets have sampling uncertainty; do not tune models using these test results.")
    st.markdown("#### Model observations on the current data")
    for name, model in models.items():
        values = comparison.loc[name]
        errors = confusion_matrix(target, model.predict(features), labels=LABELS)
        with st.expander(name):
            st.write(f"Accuracy {values['Accuracy']:.4f}; malignant recall {values['Recall']:.4f}; F1 {values['F1']:.4f}.")
            st.write(f"Missed {errors[0, 1]} of {errors[0].sum()} malignant cases and flagged {errors[1, 0]} of {errors[1].sum()} benign cases as malignant.")
            st.caption("Model assumptions and fixed settings are described in Methods and README.")
with data_tab:
    st.subheader("Every test sample, available to inspect")
    st.caption("The table includes the full current test CSV, not just a preview. Display filters below do not change evaluation metrics or model comparisons.")
    counts = target.map({0: "Malignant", 1: "Benign"}).value_counts().reindex(["Malignant", "Benign"], fill_value=0)
    st.bar_chart(counts.rename("Samples"))
    st.write(f"Malignant: {counts['Malignant']} | Benign: {counts['Benign']}")
    class_filter = st.radio("Display rows", ["All classes", "Malignant only", "Benign only"], horizontal=True)
    selected_features = st.multiselect("Features to display", list(features.columns), default=list(features.columns))
    mask = pd.Series(True, index=target.index)
    if class_filter != "All classes":
        mask = target == (0 if class_filter == "Malignant only" else 1)
    view = features.loc[mask, selected_features].copy()
    view.insert(0, "CSV row", target.index[mask] + 1)
    view["target"] = target[mask]
    view["Class"] = target[mask].map({0: "Malignant", 1: "Benign"})
    st.caption(f"Showing {len(view)} of {len(frame)} rows and {len(selected_features)} selected input features.")
    st.dataframe(view, width="stretch", hide_index=True, height=440)
    st.download_button("Download current validated test data", pd.concat([features, target], axis=1).to_csv(index=False), "current_test_data.csv", "text/csv")
    with st.expander("Feature summaries for the complete current test set"):
        st.dataframe(features.describe().T, width="stretch")
    with st.expander("Original uploaded columns"):
        st.dataframe(frame, width="stretch")
        st.caption("Extra columns are shown here but are not inputs to the models.")

with guide_tab:
    st.subheader("The experiment explained")
    st.markdown("#### Dataset and preprocessing")
    st.write("Source: UCI Breast Cancer Wisconsin (Diagnostic), loaded using scikit-learn's packaged copy. The original dataset has no missing values or duplicate rows. A stratified split with random seed 17 preserves class proportions: training has 170 malignant and 285 benign samples; test has 42 malignant and 72 benign samples.")
    st.write("The ten measured properties are radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry and fractal dimension. Each has a mean, standard-error and worst-value summary, producing 30 inputs.")
    st.write("StandardScaler is fitted only on training rows for Logistic Regression, KNN and SVM. Scaling is saved inside each pipeline. The tree and Gaussian Naive Bayes use raw measurements. Hyperparameters are fixed baselines; the test data is not used for tuning.")
    st.markdown("#### Models and settings")
    descriptions = {
        "Logistic Regression": ("Scaled linear baseline. Models a linear decision boundary in the standardized features.", "C=1; max_iter=2000; random_state=17"),
        "Decision Tree": ("A sequence of feature-based splits. Depth and minimum leaf size limit complexity.", "max_depth=5; min_samples_leaf=3; random_state=17"),
        "KNN": ("Predicts using the labels of five nearby training samples. Scaling prevents larger-valued measurements dominating distances.", "n_neighbors=5; uniform weights; Euclidean distance"),
        "Naive Bayes": ("Gaussian class-conditional distributions with a conditional-independence assumption. Correlated features can challenge that assumption.", "GaussianNB; default var_smoothing=1e-9"),
        "Support Vector Machine": ("An RBF kernel models nonlinear boundaries. Probability calibration uses training data only.", "C=1; kernel=rbf; gamma=scale; probability=True; random_state=17"),
    }
    for name, (description, settings) in descriptions.items():
        with st.expander(name):
            st.write(description)
            st.code(settings, language="text")
    st.markdown("#### Understanding the six metrics")
    definitions = {
        "Accuracy": "Fraction of all samples classified correctly. (TP + TN) / all samples.",
        "AUC": "How well malignant probabilities rank malignant cases above benign cases across thresholds. Uses probability for class 0, not predicted labels. Unavailable if only one class is present.",
        "Precision": "Of samples predicted malignant, how many were malignant? TP / (TP + FP).",
        "Recall": "Of actual malignant samples, how many were found? TP / (TP + FN).",
        "F1": "Harmonic mean of malignant precision and recall: 2TP / (2TP + FP + FN).",
        "MCC": "Correlation between true and predicted labels, using all four confusion-matrix counts. +1 is perfect, 0 means no correlation, -1 means complete disagreement. This app shows N/A on single-class test subsets.",
    }
    for name, definition in definitions.items():
        st.markdown(f"**{name}** - {definition}")
    st.caption("TP = malignant correctly identified; TN = benign correctly identified; FP = benign predicted malignant; FN = malignant predicted benign. Undefined precision, recall and F1 are reported as zero.")
    st.markdown("#### Reproduce the experiment")
    st.code("python -m pip install -r requirements.txt\npython prepare_data.py\npython train.py\npython -m unittest discover -s tests -v\npython -m streamlit run app.py", language="bash")
    with st.expander("Saved experiment environment"):
        st.write(f"Training Python version: {metadata['python']}; split seed: {metadata['seed']}.")
        st.json(metadata["packages"])
        st.caption("These are saved artifact versions, not a claim about the operating system currently serving the app.")
    with st.expander("Complete project README"):
        st.caption("The README describes the original bundled experiment. Its reported scores do not change when a different CSV is uploaded.")
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        st.markdown(readme)
        st.download_button("Download README", readme, "README.md", "text/markdown")
