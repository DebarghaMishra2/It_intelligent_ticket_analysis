
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data/raw/classification"
MODEL_PATH = ROOT / "models"
REPORT_PATH = ROOT / "reports"

MODEL_PATH.mkdir(parents=True, exist_ok=True)
REPORT_PATH.mkdir(parents=True, exist_ok=True)


# Original dataset category mapping
CATEGORY_MAP = {
    0: "Active Directory",
    1: "Computer-Services",
    2: "EOL",
    3: "Fileservice",
    4: "O365",
    5: "Software",
    6: "Support general"
}


# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

train_df = pd.read_csv(DATA_PATH / "train.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", train_df.shape)
print("Columns:", train_df.columns.tolist())


# --------------------------------------------------
# 3. PREPARE TEXT AND LABELS
# --------------------------------------------------

def prepare_data(df):

    if "text" not in df.columns:
        raise ValueError("Missing 'text' column in dataset.")

    if "label_name" in df.columns:
        labels = df["label_name"]

    elif "label" in df.columns:
        labels = df["label"].map(
            lambda x: CATEGORY_MAP.get(int(x), str(x))
            if pd.notna(x) and str(x).isdigit()
            else str(x)
        )

    else:
        raise ValueError("No valid label column found.")

    result = pd.DataFrame({
        "text": df["text"],
        "label": labels
    })

    result = result.dropna()

    result["text"] = result["text"].astype(str).str.strip()
    result["label"] = result["label"].astype(str).str.strip()

    result = result[
        (result["text"] != "") &
        (result["label"] != "")
    ]

    return result


train_df = prepare_data(train_df)

X = train_df["text"]
y = train_df["label"]

print("\nCategory distribution:")
print(y.value_counts())


# --------------------------------------------------
# 4. TRAIN / VALIDATION SPLIT
# --------------------------------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Validation records:", len(X_val))


# --------------------------------------------------
# 5. DEFINE MODELS
# --------------------------------------------------

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ),

    "Multinomial Naive Bayes": MultinomialNB(
        alpha=0.5
    ),

    "Calibrated Linear SVM": CalibratedClassifierCV(
        estimator=LinearSVC(
            class_weight="balanced"
        ),
        method="sigmoid",
        cv=3
    )
}


# --------------------------------------------------
# 6. TRAIN AND COMPARE MODELS
# --------------------------------------------------

results = {}
best_score = -1
best_model = None
best_name = None

for name, classifier in models.items():

    print("\n" + "=" * 55)
    print("Training:", name)
    print("=" * 55)

    pipeline = Pipeline([

        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                max_features=50000,
                sublinear_tf=True
            )
        ),

        ("classifier", classifier)

    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_val)

    accuracy = accuracy_score(y_val, predictions)

    macro_f1 = f1_score(
        y_val,
        predictions,
        average="macro"
    )

    weighted_f1 = f1_score(
        y_val,
        predictions,
        average="weighted"
    )

    print("Accuracy:", round(accuracy, 4))
    print("Macro F1:", round(macro_f1, 4))
    print("Weighted F1:", round(weighted_f1, 4))

    results[name] = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1)
    }

    if macro_f1 > best_score:
        best_score = macro_f1
        best_model = pipeline
        best_name = name


# --------------------------------------------------
# 7. RETRAIN BEST MODEL ON FULL TRAINING DATA
# --------------------------------------------------

print("\nBest model:", best_name)

best_model.fit(X, y)


# --------------------------------------------------
# 8. SAVE MODEL
# --------------------------------------------------

joblib.dump(
    best_model,
    MODEL_PATH / "ticket_classifier.joblib"
)

# Save model comparison
with open(
    REPORT_PATH / "model_comparison.json",
    "w"
) as file:

    json.dump(
        {
            "best_model": best_name,
            "validation_results": results
        },
        file,
        indent=4
    )


# --------------------------------------------------
# 9. FINAL MESSAGE
# --------------------------------------------------

print("\n" + "=" * 55)

print("MODEL TRAINING COMPLETED!")

print("Best model:", best_name)

print(
    "Validation Macro F1:",
    round(best_score, 4)
)

print(
    "Model saved at:",
    MODEL_PATH / "ticket_classifier.joblib"
)

print("=" * 55)
