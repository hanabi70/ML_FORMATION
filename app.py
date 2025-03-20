from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Charger le modèle
model = joblib.load('data/churn_model.pkl')

# Variables disponibles
AVAILABLE_VARS = ["Age", "Years", "Num_Sites", "Account_Manager"]

@app.route('/')
def home():
    return render_template('index.html', variables=AVAILABLE_VARS)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer les données du formulaire
        data = request.form
        
        # Créer un DataFrame avec les variables sélectionnées
        selected_vars = [var for var in AVAILABLE_VARS if var in data]
        if not selected_vars:
            return jsonify({'error': 'Aucune variable sélectionnée'}), 400
            
        input_data = pd.DataFrame([{var: float(data[var]) for var in selected_vars}])
        
        # Faire la prédiction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        return jsonify({
            'prediction': int(prediction),
            'probability': float(probability),
            'selected_vars': selected_vars
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True) 