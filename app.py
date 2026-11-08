"""Starter UI: complete saved-model inference and evaluation below."""
from pathlib import Path
import pandas as pd
import streamlit as st
from train import MODEL_FILES

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Classification Study", layout="wide")
st.title("Classification Study")
st.caption("ML Assignment 2 - learning workspace")
st.warning("Starter interface: model training and evaluation are not implemented yet.")
choice = st.sidebar.selectbox("Classifier", list(MODEL_FILES))
upload = st.sidebar.file_uploader("Upload held-out test CSV with target", type=["csv"])
bundled = ROOT / "test_data.csv"
if bundled.exists():
    st.sidebar.download_button("Download sample test CSV", bundled.read_bytes(),
                               file_name="test_data.csv", mime="text/csv")
use_sample = st.sidebar.checkbox("Preview bundled test data", value=True)
source = upload if upload is not None else bundled if use_sample and bundled.exists() else None
if source is not None:
    try:
        frame = pd.read_csv(source)
        st.write(f"{len(frame)} rows, {len(frame.columns)} columns")
        st.dataframe(frame.head(20), use_container_width=True)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        st.error(f"Unable to read CSV: {exc}")
        st.stop()
else:
    st.info("Run prepare_data.py or upload a test CSV to preview data.")

st.subheader(choice)
st.info("Next: load your saved pipeline, validate the CSV, and display evaluation results.")
# Load only trusted repository model files, never user-uploaded pickle/joblib files.
# TODO: joblib.load(ROOT / "model" / MODEL_FILES[choice]).
# TODO: validate required feature names, numeric finite values, nonempty rows,
#       target values, and missing values; reorder columns to training order.
# TODO: call evaluate_classifier; display six metrics, confusion matrix and report.
# TODO: compare all five saved models on the same uploaded test rows.
# Do not call fit() or train.main() here.
