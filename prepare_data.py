"""Prepare a reproducible learning dataset. Run from any working directory."""
import json
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent

def main():
    dataset = load_breast_cancer(as_frame=True)
    frame = dataset.frame
    audit = {"rows": len(frame), "features": frame.shape[1] - 1,
             "missing_values": int(frame.isna().sum().sum()),
             "duplicate_rows": int(frame.duplicated().sum()),
             "class_counts": frame["target"].value_counts().sort_index().to_dict()}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/dataset_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    train, test = train_test_split(
        frame, test_size=0.2, random_state=17, stratify=frame["target"]
    )
    (ROOT / "data").mkdir(exist_ok=True)
    train.to_csv(ROOT / "data" / "train_data.csv", index=False)
    test.to_csv(ROOT / "test_data.csv", index=False)
    print(f"Features: {frame.shape[1] - 1}; train: {len(train)}; test: {len(test)}")
    print("Target: 0 = malignant, 1 = benign")
    print("Keep test_data.csv held out from training and parameter selection.")

if __name__ == "__main__":
    main()
