from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np

# import flasgger
from flasgger import Swagger

app = Flask(__name__)
Swagger(app)

# Load model
with open("models/classifier_model.pkl", "rb") as f:
    classifier = pickle.load(f)


@app.route("/")
def welcome():
    """Simple welcome message."""
    return "Welcome All"


@app.route("/predict", methods=["GET"])
def predict_note_authentication():
    """
    Predict banknote authenticity (single row)
    ---
    parameters:
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
    responses:
      200:
        description: Prediction result (0=authentic, 1=forged)
        schema:
          type: object
          properties:
            prediction:
              type: integer
              example: 0
      400:
        description: Invalid input
    """
    try:
        variance = float(request.args.get("variance"))
        skewness = float(request.args.get("skewness"))
        curtosis = float(request.args.get("curtosis"))
        entropy = float(request.args.get("entropy"))

        features = np.array([[variance, skewness, curtosis, entropy]])
        prediction = classifier.predict(features)

        return jsonify({"prediction": int(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/predict_file", methods=["POST"])
def predict_note_authentication_file():
    """
    Predict banknote authenticity from CSV
    ---
    consumes:
      - multipart/form-data
    parameters:
      - name: file
        in: formData
        type: file
        required: true
    responses:
      200:
        description: Predictions for all rows
        schema:
          type: object
          properties:
            predictions:
              type: array
              items:
                type: integer
              example: [0, 1, 0, 1]
      400:
        description: Invalid input or file
    """
    try:
        # Check if a file was uploaded
        if "file" not in request.files or request.files["file"].filename == "":
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        # Ensure file pointer is at start
        file.seek(0)

        # Read CSV and predict
        df_test = pd.read_csv(file)
        prediction = classifier.predict(df_test)

        return jsonify({"predictions": prediction.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=False)
# main()
# app.run(debug=False)
