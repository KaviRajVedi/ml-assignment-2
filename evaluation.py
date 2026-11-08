"""Shared validation and metrics; malignant (0) is the positive class."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)
from config import TARGET, POSITIVE_CLASS


def validate_dataset(frame, feature_names):
    """Accept extra columns; return finite numeric inputs in training order."""
    if frame.empty:
        raise ValueError("The CSV has no data rows.")
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate column names are not allowed.")
    required = list(feature_names) + [TARGET]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    try:
        numeric = frame.loc[:, required].apply(pd.to_numeric, errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError("Features and target must contain numeric values only.") from exc
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Missing values, NaN and infinity are not allowed.")
    if not numeric[TARGET].isin([0, 1]).all():
        raise ValueError("Target must be 0 (malignant) or 1 (benign).")
    return numeric.loc[:, list(feature_names)], numeric[TARGET].astype(int)


def positive_probability(model, features):
    index = list(model.classes_).index(POSITIVE_CLASS)
    return model.predict_proba(features)[:, index]


def evaluate_classifier(model, features, target):
    predicted = model.predict(features)
    positive_target = np.asarray(target) == POSITIVE_CLASS
    auc = (float(roc_auc_score(positive_target, positive_probability(model, features)))
           if len(np.unique(target)) == 2 else None)
    return {
        "Accuracy": float(accuracy_score(target, predicted)),
        "AUC": auc,
        "Precision": float(precision_score(target, predicted, pos_label=POSITIVE_CLASS, zero_division=0)),
        "Recall": float(recall_score(target, predicted, pos_label=POSITIVE_CLASS, zero_division=0)),
        "F1": float(f1_score(target, predicted, pos_label=POSITIVE_CLASS, zero_division=0)),
        "MCC": float(matthews_corrcoef(target, predicted)) if len(np.unique(target)) == 2 else None,
    }
