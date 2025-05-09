#!/usr/bin/env python3
import sys
import os
import shutil
import time
import traceback

from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Paths and filenames
MODEL_DIR = 'model'
MODEL_FILE = os.path.join(MODEL_DIR, 'model.pkl')
MODEL_COLUMNS_FILE = os.path.join(MODEL_DIR, 'model_columns.pkl')
TRAINING_DATA = 'data/titanic.csv'
INCLUDE_COLUMNS = ['Age', 'Sex', 'Embarked', 'Survived']
TARGET_COLUMN = 'Survived'

# Ensure model directory exists and is writable
os.makedirs(MODEL_DIR, exist_ok=True)

# In-memory placeholders
clf = None
model_columns = None

@app.route('/', methods=['GET'])
def home():
    return (
        "Flask API running. Use GET /train to train, POST /predict to predict, GET /wipe to reset."
    )

@app.route('/train', methods=['GET'])
def train():
    global clf, model_columns
    try:
        # Load data
        df = pd.read_csv(TRAINING_DATA)
        df = df[INCLUDE_COLUMNS].copy()

        # Identify and handle categorical columns
        categoricals = [col for col in df if df[col].dtype == 'O']
        df[categoricals] = df[categoricals].fillna('')  # fill missing strings
        for col in df.columns.difference(categoricals):
            df[col].fillna(0, inplace=True)

        # One-hot encode
        df_ohe = pd.get_dummies(df, columns=categoricals, dummy_na=True)

        # Split features and target
        X = df_ohe.drop(columns=[TARGET_COLUMN])
        y = df_ohe[TARGET_COLUMN]

        # Store column list for prediction alignment
        model_columns = list(X.columns)
        joblib.dump(model_columns, MODEL_COLUMNS_FILE)

        # Train classifier
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier()
        start = time.time()
        clf.fit(X, y)
        elapsed = time.time() - start
        print(f'Trained in {elapsed:.1f}s — score: {clf.score(X, y):.3f}')

        # Save model
        joblib.dump(clf, MODEL_FILE)
        return 'Training completed successfully.'

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

@app.route('/predict', methods=['POST'])
def predict():
    global clf, model_columns
    if clf is None or model_columns is None:
        return jsonify({'error': 'No model found. Run GET /train first.'}), 400

    try:
        data = request.get_json(force=True)
        df = pd.DataFrame(data)
        df_ohe = pd.get_dummies(df)
        df_ohe = df_ohe.reindex(columns=model_columns, fill_value=0)

        preds = clf.predict(df_ohe)
        return jsonify({'prediction': [int(p) for p in preds]})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500

@app.route('/wipe', methods=['GET'])
def wipe():
    global clf, model_columns
    try:
        shutil.rmtree(MODEL_DIR)
        os.makedirs(MODEL_DIR, exist_ok=True)
        clf = None
        model_columns = None
        return 'Model directory wiped.'
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Determine port (default to 9999)
    try:
        port = int(sys.argv[1])
    except Exception:
        port = 9999

    # Attempt to load existing model
    try:
        clf = joblib.load(MODEL_FILE)
        model_columns = joblib.load(MODEL_COLUMNS_FILE)
        print('Loaded existing model.')
    except Exception:
        print('No existing model found; run GET /train to create one.')

    # Start Flask server
    app.run(host='0.0.0.0', port=port, debug=False)
