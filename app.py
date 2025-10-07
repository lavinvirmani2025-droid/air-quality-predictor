from flask import Flask, request, render_template
import pandas as pd
import gdown
import os
import joblib

app = Flask(__name__)

# Google Drive file ID from your link
file_id = "14sCTd_pBAWek4u8GjRxvTgyfxHXXAB-3"
model_path = "air_quality_model.pkl"

# Download the model if it doesn't exist locally
if not os.path.exists(model_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, model_path, quiet=False)

# Load the model
model = joblib.load(model_path)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            # Example: assuming your model expects these inputs
            pm25 = float(request.form["pm25"])
            pm10 = float(request.form["pm10"])
            so2 = float(request.form["so2"])
            no2 = float(request.form["no2"])
            co = float(request.form["co"])
            o3 = float(request.form["o3"])
            
            # Put inputs into a DataFrame for prediction
            input_data = pd.DataFrame([[pm25, pm10, so2, no2, co, o3]],
                                      columns=["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"])
            
            prediction = model.predict(input_data)[0]
        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
