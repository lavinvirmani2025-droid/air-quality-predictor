from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load your model
with open("air_quality_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            feature1 = float(request.form.get("feature1", 0))
            feature2 = float(request.form.get("feature2", 0))
            # Example: if your model expects more features, add them
            prediction = model.predict([[feature1, feature2]])[0]
        except Exception as e:
            prediction = f"Error: {e}"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
