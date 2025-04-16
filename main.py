from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load your models
model1 = pickle.load(open("model1.pkl", "rb"))
model2 = pickle.load(open("model2.pkl", "rb"))

# Mapping vehicle types to one-hot encoding
vehicle_mapping = {
    "bicycle": [1, 0, 0, 0],
    "scooter": [0, 1, 0, 0],
    "motorcycle": [0, 0, 1, 0],
    "electric_scooter": [0, 0, 0, 1],
}


@app.route("/")
def home():
    return render_template("index.html")  # Render your HTML page


@app.route("/predict_time", methods=["POST"])
def predict_time():
    data = request.form
    age = int(data["age"])
    distance = float(data["distance"])
    selected_vehicle = data.get("vehicle")

    vehicle_features = vehicle_mapping.get(selected_vehicle, [0, 0, 0, 0])

    features = np.array([[age, distance] + vehicle_features])
    prediction = round(model1.predict(features)[0], 2)

    return jsonify({"Predicted Time Taken (min)": prediction})


@app.route("/predict_rating", methods=["POST"])
def predict_rating():
    data = request.form
    age = int(data["age"])
    time_taken = float(data["time"])
    distance = float(data["distance"])

    vehicle_features = [
        int("vehicle_bicycle" in data),
        int("vehicle_electric" in data),
        int("vehicle_motorcycle" in data),
        int("vehicle_scooter" in data),
    ]

    features = np.array([[age, time_taken, distance] + vehicle_features])
    prediction = round(model2.predict(features)[0], 2)

    return jsonify({"Predicted Delivery Rating": prediction})


if __name__ == "__main__":
    app.run(debug=True)
