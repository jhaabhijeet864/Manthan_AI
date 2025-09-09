from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)
model = tf.keras.models.load_model('models/trained_models/fish_cnn.h5')

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

if __name__ == '__main__':
    app.run(port=5001)