from flask import Flask, request, jsonify
import joblib
import pandas as pd


# Load trained model at startup
model = joblib.load("model.joblib")

# Initialize Flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    """
    Receives a JSON object, performs a prediction, and returns the result.
    Example input:
    {
      "feature1": value,
      "feature2": value,
      ...
    }
    """
    # Parse JSON and convert to DataFrame
    data = request.get_json()
    input_df = pd.DataFrame([data])

    # Make prediction
    prediction = model.predict(input_df)[0]

    # Return result as JSON
    return jsonify({"prediction": int(prediction)})

if __name__ == "__main__":
    # Start server
    app.run(debug=True)
