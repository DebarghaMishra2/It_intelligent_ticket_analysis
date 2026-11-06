
from pathlib import Path
import joblib


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT / "models" / "ticket_classifier.joblib"
)


# ==========================================
# 2. LOAD TRAINED MODEL
# ==========================================

def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}. "
            "Run train.py first."
        )

    return joblib.load(MODEL_PATH)


# ==========================================
# 3. TICKET CLASSIFICATION
# ==========================================

def predict_ticket(description, model):

    if not isinstance(description, str):
        raise ValueError("Description must be a string.")

    if not description.strip():
        raise ValueError("Ticket description cannot be empty.")

    # Predict category
    prediction = model.predict([description])[0]

    # Get calibrated model probabilities
    probabilities = model.predict_proba([description])[0]

    # Get model category names
    classes = model.classes_

    # Rank categories by probability
    ranked_categories = sorted(
        zip(classes, probabilities),
        key=lambda item: item[1],
        reverse=True
    )

    result = {

        "ticket_description": description,

        "predicted_category": str(prediction),

        "model_probability": round(
            float(max(probabilities)), 4
        ),

        "top_categories": [

            {
                "category": str(category),

                "probability": round(
                    float(probability), 4
                )
            }

            for category, probability in ranked_categories[:3]
        ]
    }

    return result


# ==========================================
# 4. MULTI-LINE TICKET INPUT
# ==========================================

def get_ticket_description():

    print("\nEnter your IT ticket description.")

    print("Press ENTER on an empty line when finished.\n")

    lines = []

    while True:

        line = input()

        if not line.strip():
            break

        lines.append(line)

    return "\n".join(lines)


# ==========================================
# 5. MAIN FUNCTION
# ==========================================

def main():

    try:

        model = load_model()

        description = get_ticket_description()

        if not description.strip():

            print("Please enter a valid ticket description.")
            return

        result = predict_ticket(description, model)

        print("\n" + "=" * 55)

        print("IT-SENTINEL TICKET CLASSIFICATION")

        print("=" * 55)

        print(
            "\nPredicted category:",
            result["predicted_category"]
        )

        print(
            "Model probability:",
            result["model_probability"]
        )

        print("\nTop three categories:")

        for item in result["top_categories"]:

            print(
                f"{item['category']}: "
                f"{item['probability']:.4f}"
            )

        print("\nClassification completed.")

    except (ValueError, FileNotFoundError) as error:

        print("\nError:", error)


if __name__ == "__main__":

    main()
