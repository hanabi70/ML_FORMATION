import pytest
import pandas as pd
import joblib
import os
from sklearn.linear_model import LogisticRegression

def test_train_data_exists():
    assert os.path.exists('data/train_data.csv'), "Le fichier train_data.csv n'existe pas"

def test_model_training():
    # Vérifier que le modèle est créé
    assert os.path.exists('data/churn_model.pkl'), "Le modèle n'a pas été créé"
    
    # Charger le modèle et vérifier son type
    model = joblib.load('data/churn_model.pkl')
    assert isinstance(model, LogisticRegression), "Le modèle n'est pas une instance de LogisticRegression"
    
    # Vérifier que le modèle a été entraîné
    assert hasattr(model, 'coef_'), "Le modèle n'a pas été entraîné"

def test_train_data_structure():
    # Charger les données d'entraînement
    train_data = pd.read_csv('data/train_data.csv')
    
    # Vérifier les colonnes requises
    required_columns = ["Age", "Years", "Num_Sites", "Account_Manager", "Churn"]
    for col in required_columns:
        assert col in train_data.columns, f"La colonne {col} est manquante dans les données d'entraînement"
    
    # Vérifier que les données ne sont pas vides
    assert len(train_data) > 0, "Les données d'entraînement sont vides"
    
    # Vérifier qu'il n'y a pas de valeurs manquantes
    assert not train_data[required_columns].isnull().any().any(), "Il y a des valeurs manquantes dans les données" 