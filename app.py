from flask import Flask, render_template, request
import joblib
import numpy as np

# Create Flask app
app = Flask(__name__)

# Home route
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
            # Read input values from form
            pm25 = float(request.form["pm25"])
            pm10 = float(request.form["pm10"])
            no2 = float(request.form["no2"])
            so2 = float(request.form["so2"])
            co = float(request.form["co"])
            o3 = float(request.form["o3"])

            # Load model
            model = joblib.load("air_quality_model.pkl")

            # Predict AQI
            pred = model.predict(np.array([[pm25, pm10, no2, so2, co, o3]]))[0]

            # Classify AQI
            if pred <= 50:
                category = "Good"
            elif pred <= 100:
                category = "Moderate"
            elif pred <= 200:
                category = "Unhealthy"
            elif pred <= 300:
                category = "Very Unhealthy"
            else:
                category = "Hazardous"

            prediction = f"{pred:.2f} → {category}"

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html", prediction=prediction)

# Run app (for local testing)
if __name__ == "__main__":
    app.run(debug=True)
