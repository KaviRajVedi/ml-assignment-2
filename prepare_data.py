"""Prepare a reproducible learning dataset. Run from any working directory."""
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent

def main():
    dataset = load_breast_cancer(as_frame=True)
    frame = dataset.frame
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
