# ML Assignment 2 - Classification study

Learning workspace: implementation and results are still to be completed.

## a. Problem statement
Write your own explanation of the classification problem and why it matters.
Suggested project: classify breast tumour measurements as malignant or benign.
This is an educational experiment, not a clinical tool.

## b. Dataset description
Suggested source: [UCI Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic).
The dataset has 569 instances, 30 numeric input features and two classes.
The scikit-learn packaged copy is available without a separate download.
Its target encoding is 0 = malignant, 1 = benign; decide and document your positive class.

Before reporting results, document class counts, missing values, duplicates,
feature meanings, split proportions, random seed, and preprocessing.
Use one stratified train/test split for all five models. Fit preprocessing only
on the training partition, preferably inside a saved sklearn Pipeline.
Do not tune hyperparameters using the held-out test set.

## c. GitHub Repository Link
TODO: add your own repository URL after publishing.

## d. Models used
Fill this table with measured results from your own execution.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | TODO | TODO | TODO | TODO | TODO | TODO |
| Decision Tree | TODO | TODO | TODO | TODO | TODO | TODO |
| KNN | TODO | TODO | TODO | TODO | TODO | TODO |
| Naive Bayes | TODO | TODO | TODO | TODO | TODO | TODO |
| Support Vector Machine | TODO | TODO | TODO | TODO | TODO | TODO |

State your positive class and averaging method. Compute AUC from continuous
scores or probabilities, not predicted class labels. AUC is undefined when
an uploaded test set contains only one class; show N/A with an explanation.

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | TODO: support your interpretation with measured results. |
| Decision Tree | TODO: compare train/test behaviour and tree complexity. |
| KNN | TODO: discuss scaling, neighbours, and class errors. |
| Naive Bayes | TODO: inspect correlated features and model assumptions. |
| Support Vector Machine | TODO: discuss kernel choice and class errors. |
| Overall winner for your dataset | TODO: justify using an appropriate metric and limitations. |

## Setup
Use Python 3.11 or 3.12 for a broadly supported deployment environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python prepare_data.py
# Complete train.py before running it:
python train.py
python -m streamlit run app.py
```

If PowerShell activation is blocked, run `.venv\Scripts\python.exe` directly.
Preparation creates the held-out CSV; training is deliberately left as a learning task.
The initial app previews data and lists the required models; evaluation is unfinished.

## Files
- [prepare_data.py](prepare_data.py): reproducible dataset split.
- [train.py](train.py): implementation prompts for all five models.
- [evaluation.py](evaluation.py): shared evaluation interface to implement.
- [app.py](app.py): initial Streamlit interface.
- [model/](model/): store all five trained pipelines here and commit them.
- [results/](results/): measured comparisons and classification reports.
- [docs/work-plan.md](docs/work-plan.md): implementation and verification steps.
- [docs/submission-checklist.md](docs/submission-checklist.md): deployment and submission requirements.

## Live app
TODO: add your own Streamlit Community Cloud URL.

## Reproducibility
After successful training, record Python and dependency versions and pin the
versions used to save and load your models. Execute the work on BITS Virtual Lab
and capture your own screenshot. Explain and adapt the starter yourself; do not
submit an unchanged template or another student's work.
