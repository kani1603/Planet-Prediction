from flask import Flask, render_template, request
from flask_cors import CORS   # <-- import CORS here
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # <-- enable CORS for all routes

# Load model and encoders
model = joblib.load("model.pkl")
le_interests = joblib.load("le_interests.pkl")
le_skill = joblib.load("le_skill.pkl")
le_activity = joblib.load("le_activity.pkl")
le_target = joblib.load("le_target.pkl")
feature_columns = joblib.load("feature_columns.pkl")

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        interests = request.form["interests"]
        skill = request.form["skill"]
        past_activity = request.form["past_activity"]

        # Encode inputs
        interests_enc = le_interests.transform([interests])[0]
        skill_enc = le_skill.transform([skill])[0]
        activity_enc = le_activity.transform([past_activity])[0]

        # Prepare input array in the correct feature order
        X = np.array([[interests_enc, skill_enc, activity_enc]])

        # Predict class index
        pred_enc = model.predict(X)[0]
        prediction = le_target.inverse_transform([pred_enc])[0]

        # Predict probabilities
        probas = model.predict_proba(X)[0]
        classes = le_target.classes_
        probabilities = {cls: round(prob * 100, 2) for cls, prob in zip(classes, probas)}

        return render_template("index.html", prediction=prediction, probabilities=probabilities)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
