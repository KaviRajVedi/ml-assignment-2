"""Regression checks for data handling, metrics and saved-model UI behaviour."""
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, PropertyMock
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import MODEL_FILES
from evaluation import validate_dataset, evaluate_classifier


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({"a": [1, 2], "b": [3, 4], "target": [0, 1]})

    def test_reorder_and_extra_columns(self):
        x, y = validate_dataset(self.frame.assign(extra="ignored")[["target", "b", "extra", "a"]], ["a", "b"])
        self.assertEqual(list(x.columns), ["a", "b"])
        self.assertEqual(y.tolist(), [0, 1])

    def test_invalid_inputs(self):
        cases = [self.frame.iloc[:0], self.frame.drop(columns="a"),
                 self.frame.assign(a=[np.nan, 1]), self.frame.assign(a=[np.inf, 1]),
                 self.frame.assign(a=["bad", "1"]), self.frame.assign(target=[0, 2]),
                 self.frame.assign(target=[0, 0.5]), self.frame.assign(target=[0, None])]
        for frame in cases:
            with self.subTest(frame=frame), self.assertRaises(ValueError):
                validate_dataset(frame, ["a", "b"])

    def test_probability_orientation(self):
        class Model:
            classes_ = np.array([0, 1])
            def predict(self, x):
                return np.array([0, 1, 0, 1])
            def predict_proba(self, x):
                return np.array([[.9, .1], [.2, .8], [.8, .2], [.1, .9]])
        metrics = evaluate_classifier(Model(), None, np.array([0, 1, 0, 1]))
        self.assertTrue(all(value == 1 for value in metrics.values()))


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = pd.read_csv(ROOT / "test_data.csv")
        cls.models = {name: joblib.load(ROOT / "model" / file) for name, file in MODEL_FILES.items()}

    def test_persisted_models_match_recorded_metrics(self):
        recorded = pd.read_csv(ROOT / "results/metrics.csv").set_index("ML Model Name")
        for name, model in self.models.items():
            x, y = validate_dataset(self.frame, model.feature_names_in_)
            actual = evaluate_classifier(model, x, y)
            for key, value in actual.items():
                self.assertAlmostEqual(value, recorded.loc[name, key], places=12)

    def test_single_class_subset(self):
        model = next(iter(self.models.values()))
        for label in [0, 1]:
            x, y = validate_dataset(self.frame[self.frame.target == label], model.feature_names_in_)
            metrics = evaluate_classifier(model, x, y)
            self.assertIsNone(metrics["AUC"])
            self.assertIsNone(metrics["MCC"])

    def test_scaler_fitted_only_on_training_rows(self):
        train = pd.read_csv(ROOT / "data/train_data.csv").drop(columns="target")
        for name in ["Logistic Regression", "KNN", "Support Vector Machine"]:
            scaler = self.models[name].steps[0][1]
            self.assertEqual(scaler.n_samples_seen_, len(train))
            np.testing.assert_allclose(scaler.mean_, train.mean().to_numpy())

    def test_app_model_selection_and_metrics(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        for name in MODEL_FILES:
            app.sidebar.selectbox[0].select(name).run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.metric), 6)
            x, y = validate_dataset(self.frame, self.models[name].feature_names_in_)
            expected = evaluate_classifier(self.models[name], x, y)
            self.assertEqual([item.value for item in app.metric], [f"{v:.4f}" for v in expected.values()])
        app.sidebar.checkbox[0].uncheck().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric), 0)

    def test_explorer_filters_do_not_change_evaluation(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        scores = [item.value for item in app.metric]
        self.assertEqual(len(app.tabs), 5)
        app.radio[0].set_value("Malignant only").run()
        self.assertFalse(app.exception)
        self.assertEqual([item.value for item in app.metric], scores)
        table = next(item.value for item in app.dataframe if "Class" in item.value.columns)
        self.assertEqual(len(table), 42)
        self.assertTrue((table.target == 0).all())
        app.multiselect[0].set_value([]).run()
        self.assertFalse(app.exception)
        table = next(item.value for item in app.dataframe if "Class" in item.value.columns)
        self.assertEqual(list(table.columns), ["CSV row", "target", "Class"])
        app.checkbox(key="only_errors").check().run()
        self.assertFalse(app.exception)
        mistakes = next(item.value for item in app.dataframe if "Predicted class" in item.value.columns)
        self.assertEqual(len(mistakes), 2)
        self.assertEqual([item.value for item in app.metric], scores)

    def test_app_without_pandas_styler(self):
        from streamlit.testing.v1 import AppTest
        for label in [None, 1]:
            frame = self.frame if label is None else self.frame[self.frame.target == label]
            with self.subTest(label=label), patch.object(
                pd.DataFrame, "style", new_callable=PropertyMock,
                side_effect=AttributeError("The '.style' accessor requires jinja2")
            ), patch("streamlit.delta_generator.DeltaGenerator.file_uploader",
                     return_value=io.BytesIO(frame.to_csv(index=False).encode())):
                app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
                self.assertFalse(app.exception)
                self.assertFalse(app.error)
                table = next(item.value for item in app.dataframe if item.value.shape == (5, 6) and "AUC" in item.value.columns)
                self.assertEqual(table.shape, (5, 6))
                if label is not None:
                    self.assertTrue((table["AUC"] == "N/A").all())
                else:
                    self.assertEqual(table.loc["Support Vector Machine", "Accuracy"], "1.0000")

    def test_app_upload_validation_and_single_class(self):
        from streamlit.testing.v1 import AppTest
        uploads = [(b"a,a,target\n1,2,0", "Duplicate"),
                   (b"a,target\n1,0", "Missing"),
                   (b"", "Invalid")]
        for data, expected in uploads:
            with self.subTest(expected=expected), patch("streamlit.delta_generator.DeltaGenerator.file_uploader", return_value=io.BytesIO(data)):
                app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
                self.assertFalse(app.exception)
                self.assertTrue(any(expected in item.value for item in app.error))
        data = self.frame[self.frame.target == 1].to_csv(index=False).encode()
        with patch("streamlit.delta_generator.DeltaGenerator.file_uploader", return_value=io.BytesIO(data)):
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertEqual(app.metric[1].value, "N/A")
            self.assertEqual(app.metric[5].value, "N/A")


if __name__ == "__main__":
    unittest.main()
