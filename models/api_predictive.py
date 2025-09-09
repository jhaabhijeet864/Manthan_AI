from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('models/trained_models/species_rf.pkl')

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

if __name__ == '__main__':
    app.run(port=5002)