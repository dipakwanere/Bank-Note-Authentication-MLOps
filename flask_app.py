from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load model
with open("models/classifier_model.pkl", "rb") as f:
    classifier = pickle.load(f)


@app.route("/")
def welcome():
    return "Welcome All"


@app.route("/predict", methods=["GET"])
def predict_note_authentication():
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
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        df_test = pd.read_csv(file)

        prediction = classifier.predict(df_test)

        return jsonify({"predictions": prediction.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
