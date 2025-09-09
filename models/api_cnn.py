from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Resolve model path relative to this file to avoid CWD issues
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'trained_models', 'fish_cnn.h5')
model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(img):
    img = img.resize((224, 224))  # adjust to your model's input size
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/predict_species', methods=['POST'])
def predict_species():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    img = Image.open(request.files['file'])
    x = preprocess_image(img)
    pred = model.predict(x)[0][0]
    label = 'positive' if pred > 0.5 else 'negative'
    return jsonify({'species': label, 'confidence': float(pred)})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, threaded=True)