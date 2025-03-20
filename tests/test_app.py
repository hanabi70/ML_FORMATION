import pytest
from app import app
import pandas as pd
import joblib
import os

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_page(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b'Age' in response.data
    assert b'Years' in response.data
    assert b'Num_Sites' in response.data
    assert b'Account_Manager' in response.data

def test_predict_with_valid_data(client):
    test_data = {
        'Age': '30',
        'Years': '5',
        'Num_Sites': '3',
        'Account_Manager': '1'
    }
    response = client.post('/predict', data=test_data)
    assert response.status_code == 200
    data = response.get_json()
    assert 'prediction' in data
    assert 'probability' in data
    assert 'selected_vars' in data

def test_predict_with_invalid_data(client):
    test_data = {
        'Age': 'invalid',
        'Years': '5'
    }
    response = client.post('/predict', data=test_data)
    assert response.status_code == 400

def test_predict_with_no_data(client):
    response = client.post('/predict', data={})
    assert response.status_code == 400
    data = response.get_json()
    assert 'error' in data
    assert data['error'] == 'Aucune variable sélectionnée' 