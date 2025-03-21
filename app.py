from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
from prometheus_client import Summary, Counter
from datetime import datetime
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')
API_CALLS = Counter('api_calls_total', 'Total number of API calls', ['endpoint'])

# Liste pour stocker les appels API avec leur timestamp
api_calls_history = []

app = Flask(__name__)
# Charger le modèle
model = joblib.load('data/churn_model.pkl')

# Variables disponibles
AVAILABLE_VARS = ["Age", "Years", "Num_Sites", "Account_Manager"]

@app.route('/')
def home():
    API_CALLS.labels(endpoint='home').inc()
    api_calls_history.append({'endpoint': 'home', 'timestamp': datetime.now()})
    return render_template('index.html', variables=AVAILABLE_VARS)

@app.route('/get_graphs_data')
def get_graphs_data():
    # Créer le graphique des appels API
    api_calls_data = None
    if api_calls_history:
        df_calls = pd.DataFrame(api_calls_history)
        df_calls['minute'] = df_calls['timestamp'].dt.floor('min')
        calls_by_minute = df_calls.groupby(['minute', 'endpoint']).size().reset_index(name='count')
        api_calls_data = calls_by_minute.to_dict('records')
    
    # Créer le graphique des churns
    df = pd.read_csv('data/train_data.csv')
    churn_counts = df['Churn'].value_counts()
    churn_data = pd.DataFrame({
        'Status': ['Non Churn', 'Churn'],
        'Count': [churn_counts[0], churn_counts[1]]
    }).to_dict('records')
    
    return jsonify({
        'api_calls': api_calls_data,
        'churn_data': churn_data
    })

@REQUEST_TIME.time()
@app.route('/predict', methods=['POST'])
def predict():
    API_CALLS.labels(endpoint='predict').inc()
    api_calls_history.append({'endpoint': 'predict', 'timestamp': datetime.now()})
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
    app.run(host='0.0.0.0', port=5000,debug=True) 