# filepath: d:\Coding Journey\Hackathons\Dronathon\Prototype_Mak1\models\quick_test.py
import joblib  # or pickle
import numpy as np

# Load your model
model = joblib.load('data/trained_models/your_model.pkl')

# Example input (replace with your feature values)
sample = np.array([[feature1, feature2, ...]])

# Predict
prediction = model.predict(sample)
print("Prediction:", prediction)