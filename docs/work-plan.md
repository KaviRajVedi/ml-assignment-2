# Git and VM workflow

## On your computer
Review the changes and measured results, then commit and push:

```bash
git status
git add .
git commit -m "Implement classifier training and Streamlit evaluation"
git push origin main
```

The environment directory is ignored; model files and results are intentionally tracked.

## In BITS Virtual Lab
Stop the old Streamlit process with Ctrl+C. From your cloned repository:

```bash
git status
git pull --ff-only
python -m pip install -r requirements.txt
python prepare_data.py
python train.py
python -m unittest discover -s tests -v
python -m streamlit run app.py
```

If Git reports local changes that would be overwritten, preserve them in a commit
or stash before pulling. Do not reset or discard your work blindly.
Open http://localhost:8501, select each classifier, inspect comparison results,
and upload the bundled test CSV to verify the same metrics. Capture genuine lab
execution evidence. Running training updates model artifacts, results and the README;
commit and push those lab outputs if they are the versions you deploy and submit.

## Understanding the implementation
- Scaling is learned from training rows only and travels with each saved model.
- Stratification preserves class proportions; test rows never enter fitting.
- Malignant (0) is positive, so AUC uses probability for class 0.
- A false negative is an actual malignant sample predicted benign.
- The F1-based winner is specific to this split; no clinical claims are made.
- Model persistence requires matching versions; the app verifies numerical dependencies.

Finally deploy to Streamlit Community Cloud and finish the submission checklist.
