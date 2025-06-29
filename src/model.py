import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import joblib

def train_and_save_model(csv_path, output_model="model.joblib"):
    """
    Loads processed data, trains an XGBoost model, and saves it to disk.
    """
    # Load the cleaned dataset
    df = pd.read_csv(csv_path)

    # Split features and label
    X = df.drop("label", axis=1)
    y = df["label"]

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)

    # Save the model to a file
    joblib.dump(model, output_model)
    print(f"[✓] Trained model saved to {output_model}")

if __name__ == "__main__":
    train_and_save_model("data/processed/cleaned_data.csv")
