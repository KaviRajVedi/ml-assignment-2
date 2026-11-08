# Work plan

1. Run dataset preparation and inspect the feature names, class balance and split.
2. Explain the problem and dataset in your own README wording.
3. Implement evaluation.py. Decide whether malignant (0) or benign (1) is positive;
   probability-column selection must match that choice.
4. Implement train.py. Save preprocessing with each estimator as one pipeline.
5. Run training. Fill the README comparison using actual results. Explain errors
   using the classification reports; avoid assuming why a model failed without evidence.
6. Complete app.py: validation, saved-model loading, six metrics, confusion matrix
   or classification report, and a comparison of the five models.
7. Verify that app metrics on test_data.csv match the training evaluation.
8. Try reordered columns, missing features, strings, NaN/infinity, empty CSV,
   unknown labels, and single-class test data. Show actionable validation messages.
9. Repeat execution on BITS Virtual Lab and capture genuine execution evidence.
10. Pin the validated dependency versions, commit your work, and deploy.

## Concepts to be ready to explain
- Why scaling is fitted only on training data.
- Why stratification helps preserve class proportions.
- Why AUC uses scores rather than predicted labels.
- What a false negative means under your chosen positive class.
- Why a single split cannot establish a universally best model.

## Git workflow
The local repository is initialized on main with no remote and no commits.

```powershell
git status
git add .
git commit -m "Add dataset preparation and project learning scaffold"
# Create an empty repository in your GitHub account, then replace YOUR_USER:
git remote add origin https://github.com/YOUR_USER/ml-assignment-2.git
git push -u origin main
```

Make meaningful commits as you implement and understand each stage.
Do not fabricate historical commits. Do not commit credentials or another
student's PDF. Saved models and test_data.csv must be committed for deployment.
