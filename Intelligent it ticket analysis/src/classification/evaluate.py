

from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models" / "ticket_classifier.joblib"
REPORT_DIR = ROOT / "reports"

# Support either original or normalized test data
possible_paths = [
    ROOT / "data/raw/classification/test.csv",
    ROOT / "data/processed/classification_test.csv"
]

TEST_PATH = next(
    (p for p in possible_paths if p.exists()),
    None
)

if TEST_PATH is None:
    raise FileNotFoundError("Test dataset not found.")

REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(TEST_PATH)

# Detect text column
text_col = (
    "text" if "text" in df.columns
    else "description"
)

# Detect category column
label_col = (
    "label_name" if "label_name" in df.columns
    else "category" if "category" in df.columns
    else "label"
)

df = df.dropna(subset=[text_col, label_col])

X_test = df[text_col].astype(str)
y_test = df[label_col].astype(str)

# Load trained model
model = joblib.load(MODEL_PATH)

# Predict
predictions = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, predictions)
macro_f1 = f1_score(
    y_test, predictions, average="macro"
)
weighted_f1 = f1_score(
    y_test, predictions, average="weighted"
)

print("\nFINAL TEST RESULTS")
print("=" * 50)

print("Test records:", len(X_test))
print("Accuracy:", round(accuracy, 4))
print("Macro F1:", round(macro_f1, 4))
print("Weighted F1:", round(weighted_f1, 4))

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)

# Confusion matrix
labels = sorted(set(y_test) | set(predictions))

cm = confusion_matrix(
    y_test,
    predictions,
    labels=labels
)

print("\nConfusion Matrix:")

print(pd.DataFrame(
    cm,
    index=labels,
    columns=labels
))

# Save predictions
pd.DataFrame({
    "actual_category": y_test,
    "predicted_category": predictions
}).to_csv(
    REPORT_DIR / "test_predictions.csv",
    index=False
)

# Save evaluation metrics
results = {
    "model": "Calibrated Linear SVM",
    "accuracy": float(accuracy),
    "macro_f1": float(macro_f1),
    "weighted_f1": float(weighted_f1)
}

with open(
    REPORT_DIR / "classification_metrics.json",
    "w"
) as f:
    json.dump(results, f, indent=4)

print("\nEvaluation completed!")
