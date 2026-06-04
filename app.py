from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import os

DATA_FILE = 'chemical_reactions.csv'

app = Flask(__name__)
CORS(app)

# load trained model
model = joblib.load('reaction_model.pkl')

data = pd.read_csv('chemical_reactions.csv')
X = data[['temperature (Kel)', 'pressure (atm)', "concentration (mol/L)", 'catalyst_type']]
y = data['success']

@app.route("/add_reaction", methods=['POST'])
def add_reaction():

    data = request.get_json()

    # load the data to csv and create new
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        # Extract numeric part from CHEM ID (e.g., "CHEM-500" -> 500)
        last_id = df['CHEM ID'].str.extract('(\d+)').iloc[-1, 0]
        new_id = int(last_id) + 1
        new_id_str = f"CHEM-{new_id}"
    else: 
        df = pd.DataFrame(columns=['CHEM ID', 'temperature (Kel)', 'pressure (atm)', 'concentration (mol/L)', 'catalyst_type', 'success'])
        new_id = 1
        new_id_str = f"CHEM-{new_id}"

        # adding a new row
    new_reaction = {
        "CHEM ID": new_id_str,
        "temperature (Kel)": data["temperature"],
        "pressure (atm)": data["pressure"],
        "concentration (mol/L)": data["concentration"],
        "catalyst_type": data["catalyst_type"],
        "success": data["success"]
    }

    # append the new reaction to the dataframe and save to csv
    df = pd.concat([df, pd.DataFrame([new_reaction])], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)

    return jsonify({"message": "Reaction added successfully", "CHEM_ID": new_id_str})

@app.route("/retrain", methods=['POST'])
def retrain():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, 'reaction_model.pkl')

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "precision" : precision_score(y_test, y_pred),
        "accuracy" : accuracy_score(y_test, y_pred),
        "f1" : f1_score(y_test, y_pred),
        "recall" : recall_score(y_test, y_pred)
    }

    # for pr-curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    pr_curve = {
        "precision" : precision.tolist(),
        "recall" : recall.tolist()
    }

    # for the confusion matrix
    cm = confusion_matrix(y_test, y_pred).tolist()

    return jsonify({ "metrics" : metrics, "confusion" : cm, "prcurve" : pr_curve })



with open("model_results.json") as f:
    results = json.load(f)

# metrics = {
#     "precision": 0.81,
#     "recall": 0.67,
#     "f1": 0.73,
#     "accuracy": 0.67
# }

# confusion_matrix = [[31, 16], [34, 69]]

# pr_curve = {
#     "precision": [0.9, 0.8, 0.7],
#     "recall": [0.1, 0.5, 0.9]
# }


@app.route("/metrics", methods=['GET'])
def get_metrics():
    return jsonify(results['metrics'])

@app.route("/confusion", methods=['GET'])
def get_confusion():
    return jsonify(results['confusion'])

@app.route("/prcurve", methods=['GET'])
def get_prcurve():
    return jsonify(results['prcurve'])

@app.route("/predict", methods=['POST'])
def predict():
    data = request.get_json(force=True)
    print(data)

    features = pd.DataFrame([[
        data["temperature"],
        data["pressure"],
        data["concentration"],
        data["catalyst_type"]

    ]], columns=['temperature (Kel)', 'pressure (atm)', 'concentration (mol/L)', 'catalyst_type'])

    prob = model.predict_proba(features)[0,1]
    prediction = int(model.predict(features)[0])
    return jsonify({"prediction": prediction, 
    "probability": float(prob)})

if __name__ == "__main__":
    app.run(debug=True)
