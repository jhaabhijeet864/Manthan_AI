# filepath: ManthanAI/scripts/predictive_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load and preprocess data
data = pd.read_csv('data/oceanographic.csv')
X = data[['temperature', 'salinity', 'chlorophyll']]  # Features
y = data['future_temperature']  # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate and save
predictions = model.predict(X_test)
print('MSE:', mean_squared_error(y_test, predictions))
import joblib
joblib.dump(model, '../models/predictive_model.pkl')