"""Implement once, then reuse in training and the app."""

def evaluate_classifier(model, features, target):
    """Return Accuracy, AUC, Precision, Recall, F1 and MCC as a dictionary.

    Learning steps:
    1. Obtain predicted labels with model.predict(features).
    2. Choose a positive class and document it in README.
    3. Obtain its probability column using model.classes_ (or use an
       appropriately oriented decision_function score).
    4. Calculate the six sklearn.metrics values with consistent conventions.
    5. Return AUC as None if target contains only one class.
    6. Handle undefined precision/recall explicitly, e.g. zero_division=0.
    """
    raise NotImplementedError("Implement the six evaluation metrics here.")
