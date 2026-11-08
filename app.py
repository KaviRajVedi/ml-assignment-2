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

st.set_page_config(page_title="Tumour Classification Lab", page_icon="??", layout="wide")
st.title("Tumour Classification Lab")
st.caption("Wisconsin Diagnostic dataset ? Five classifiers ? One held-out evaluation")
st.info("Educational model comparison. Positive class: malignant (0); benign: 1. Not for clinical use.")
choice = st.sidebar.selectbox("Classifier", list(MODEL_FILES))
upload = st.sidebar.file_uploader("Upload test CSV (features + target)", type=["csv"])
use_sample = st.sidebar.checkbox("Use bundled test data when no CSV is uploaded", value=True)
bundled = ROOT / "test_data.csv"
if bundled.exists():
    st.sidebar.download_button("Download test CSV", bundled.read_bytes(), "test_data.csv", "text/csv")
st.sidebar.caption("Uploads take priority. All models are compared on the same selected rows.")

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
evaluation_tab, comparison_tab, data_tab = st.tabs(["Selected model", "Compare all five", "Test data"])
with evaluation_tab:
    st.subheader(choice)
    for column, (metric, value) in zip(st.columns(6), comparison.loc[choice].items()):
        column.metric(metric, "N/A" if pd.isna(value) else f"{value:.4f}")
    predictions = models[choice].predict(features)
    matrix = confusion_matrix(target, predictions, labels=LABELS)
    st.subheader("Confusion matrix")
    st.caption("Rows = actual class; columns = predicted class. Malignant ? Benign is a missed malignant case.")
    st.dataframe(pd.DataFrame(matrix, index=["Actual " + n for n in CLASS_NAMES],
                             columns=["Predicted " + n for n in CLASS_NAMES]), width="stretch")
    st.subheader("Classification report")
    report = classification_report(target, predictions, labels=LABELS,
                                   target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T, width="stretch")
    output = features.copy()
    output["target"] = target
    output["prediction"] = predictions
    output["malignant_probability"] = positive_probability(models[choice], features)
    st.download_button("Download predictions", output.to_csv(index=False), "predictions.csv", "text/csv")
with comparison_tab:
    st.dataframe(comparison.style.format("{:.4f}", na_rep="N/A"), width="stretch")
    metric = st.selectbox("Metric to compare", list(comparison.columns), index=4)
    if comparison[metric].notna().any():
        st.bar_chart(comparison[[metric]])
    else:
        st.info(f"{metric} requires both classes in the uploaded data.")
    st.download_button("Download comparison", comparison.to_csv(), "comparison.csv", "text/csv")
with data_tab:
    st.write("Class counts", target.map({0: "Malignant", 1: "Benign"}).value_counts())
    st.dataframe(frame, width="stretch")
    st.dataframe(features.describe().T, width="stretch")
    st.caption("Additional CSV columns are ignored. Required features are reordered to the training schema.")
