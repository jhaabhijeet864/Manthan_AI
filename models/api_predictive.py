from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Resolve model path relative to this file to avoid CWD issues
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'trained_models', 'species_rf.pkl')
model = joblib.load(MODEL_PATH)

@app.route('/predict_trend', methods=['POST'])
def predict_trend():
    data = request.get_json()
    features = np.array([[
        data.get('temperature', 0),
        data.get('salinity', 0),
        data.get('depth', 0)
    ]])
    pred = model.predict(features)
    return jsonify({'prediction': pred[0]})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, threaded=True)