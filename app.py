from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("traffic_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    hour = float(request.form["hour"])
    temperature = float(request.form["temperature"])
    humidity = float(request.form["humidity"])
    rainfall = float(request.form["rainfall"])
    speed_limit = float(request.form["speed_limit"])
    traffic_density = float(request.form["traffic_density"])

    features = np.array([[hour,
                          temperature,
                          humidity,
                          rainfall,
                          speed_limit,
                          traffic_density]])

    prediction = model.predict(features)[0]

    return render_template(
        "index.html",
        prediction_text=f"Predicted Vehicle Count : {prediction:.0f}"
    )


if __name__ == "__main__":
    app.run(debug=True)