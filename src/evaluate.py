import pandas as pd
import joblib
from sklearn.metrics import classification_report

def evaluate_model(model_path, data_path):
    """
    Loads model and test data, prints evaluation metrics.
    """
    # Load trained model
    model = joblib.load(model_path)

    # Load processed dataset
    df = pd.read_csv(data_path)

    # Separate features and labels
    X = df.drop("label", axis=1)
    y = df["label"]

    # Predict on the full dataset
    y_pred = model.predict(X)

    # Print performance metrics
    print("[✓] Model Evaluation Report:\n")
    print(classification_report(y, y_pred))

if __name__ == "__main__":
    evaluate_model("model.joblib", "data/processed/cleaned_data.csv")
