

from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from flask_cors import CORS
import traceback

app = Flask(__name__)
CORS(app)

# Load models
logistic_model = joblib.load('models/logistic_model.pkl')
random_forest_model = joblib.load('models/random_forest_model.pkl')
xgboost_model = joblib.load('models/xgboost_model.pkl')

# Load the scaler
scaler = joblib.load('models/scaler.pkl')

# Model mapping
models = {
    "Logistic Regression": logistic_model,
    "Random Forest": random_forest_model,
    "XGBoost": xgboost_model,
}

# Define the correct feature order
all_columns = [
    'CreditScore', 
    'Gender', 
    'Age', 
    'Tenure', 
    'Balance', 
    'NumOfProducts',
    'HasCrCard', 
    'IsActiveMember', 
    'EstimatedSalary', 
    'Geography_Germany',
    'Geography_Spain'
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse request data
        data = request.json
        raw_features = data['features']

        # Encode Geography
        geography_encoded = [0, 0]  # Default: France
        if raw_features['Geography'] == 'Germany':
            geography_encoded[0] = 1
        elif raw_features['Geography'] == 'Spain':
            geography_encoded[1] = 1

        # Encode Gender
        gender_encoded = 1 if raw_features['Gender'] == 'Male' else 0

        # Create a DataFrame for all features in the correct order
        all_features = pd.DataFrame([{
            'CreditScore': float(raw_features['CreditScore']),
            'Gender': gender_encoded,
            'Age': int(raw_features['Age']),
            'Tenure': int(raw_features['Tenure']),
            'Balance': float(raw_features['Balance']),
            'NumOfProducts': int(raw_features['NumOfProducts']),
            'HasCrCard': int(raw_features['HasCrCard']),
            'IsActiveMember': int(raw_features['IsActiveMember']),
            'EstimatedSalary': float(raw_features['EstimatedSalary']),
            'Geography_Germany': geography_encoded[0],
            'Geography_Spain': geography_encoded[1]
        }], columns=all_columns)  # Ensure column order matches training

        # Scale features
        scaled_features = scaler.transform(all_features)

        # Convert scaled features to DataFrame for prediction
        final_df = pd.DataFrame(scaled_features, columns=all_columns)

        # Predict with the selected model
        selected_model = data['model']
        model = models[selected_model]
        prediction = model.predict(final_df).tolist()
       # Interpret the prediction
        if prediction[0] == 0:
            result = "The customer is not likely to churn."
        elif prediction[0] == 1:
            result = "The customer is likely to churn."

        return jsonify({"prediction": prediction[0], "message": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500




if __name__ == '__main__':
    app.run(debug=True)





