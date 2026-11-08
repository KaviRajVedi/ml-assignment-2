| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.9825 | 0.9957 | 0.9762 | 0.9762 | 0.9762 | 0.9623 |
| Decision Tree | 0.9912 | 0.9929 | 0.9767 | 1.0000 | 0.9882 | 0.9814 |
| KNN | 0.9825 | 1.0000 | 1.0000 | 0.9524 | 0.9756 | 0.9626 |
| Naive Bayes | 0.9825 | 0.9987 | 1.0000 | 0.9524 | 0.9756 | 0.9626 |
| Support Vector Machine | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Test F1 0.9762, malignant recall 0.9762, AUC 0.9957. Missed 1 of 42 malignant cases and flagged 1 benign cases as malignant. Train/test accuracy 0.9846/0.9825. A scaled linear baseline; nonlinear relationships are not explicitly represented. |
| Decision Tree | Test F1 0.9882, malignant recall 1.0000, AUC 0.9929. Missed 0 of 42 malignant cases and flagged 1 benign cases as malignant. Train/test accuracy 0.9802/0.9912. Depth is capped at 5 and leaves need 3 samples to limit complexity; probabilities are leaf proportions. |
| KNN | Test F1 0.9756, malignant recall 0.9524, AUC 1.0000. Missed 2 of 42 malignant cases and flagged 0 benign cases as malignant. Train/test accuracy 0.9758/0.9825. Uses five neighbours in standardized feature space; all features contribute to distance. |
| Naive Bayes | Test F1 0.9756, malignant recall 0.9524, AUC 0.9987. Missed 2 of 42 malignant cases and flagged 0 benign cases as malignant. Train/test accuracy 0.9297/0.9825. Uses raw measurements and class-conditional Gaussian distributions with an independence assumption. |
| Support Vector Machine | Test F1 1.0000, malignant recall 1.0000, AUC 1.0000. Missed 0 of 42 malignant cases and flagged 0 benign cases as malignant. Train/test accuracy 0.9846/1.0000. Uses standardized features and an RBF kernel; probabilities use training-only internal calibration. |
| Overall winner for this split | Support Vector Machine by malignant-class F1 (1.0000); ties are broken by recall then AUC. This is a descriptive comparison on one small held-out split, not evidence of clinical suitability or a universally best model. |
