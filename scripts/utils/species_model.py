import json
import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

# Define paths
PROCESSED_DATA_PATH = os.path.join(project_root, 'data/processed_data/fused_occurrence_oceanographic.csv')
MODEL_OUTPUT_PATH = os.path.join(project_root, 'models/trained_models/species_rf.pkl')
METRICS_OUTPUT_PATH = os.path.join(project_root, 'models/trained_models/species_rf_metrics.json')
PLOT_OUTPUT_PATH = os.path.join(project_root, 'models/trained_models/species_distribution.png')
CONFUSION_MATRIX_PATH = os.path.join(project_root, 'models/trained_models/confusion_matrix.png')

# Ensure directories exist
os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)


def load_and_prepare_data(csv_path, top_n_species=20, test_size=0.2, random_state=42):
    """
    Load the fused dataset and prepare it for training a species classifier.
    
    Args:
        csv_path: Path to the fused CSV file
        top_n_species: Number of most frequent species to keep as individual classes
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test, label_encoder, feature_names
    """
    print("✅ Loading data...")
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} records")

    # Filter to records with scientific name
    df = df[df['scientific_name'].notna()]
    print(f"✅ Records with scientific name: {len(df)}")

    # Get top N species by frequency
    species_counts = df['scientific_name'].value_counts()
    top_species = species_counts.nlargest(top_n_species).index.tolist()

    # Map rare species to "OTHER"
    df['target_species'] = df['scientific_name'].apply(
        lambda x: x if x in top_species else "OTHER"
    )

    # Extract date features
    if 'timestamp' in df.columns and df['timestamp'].notna().any():
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['day_of_year'] = df['timestamp'].dt.dayofyear
    else:
        df['day_of_year'] = 1

    # Select features
    feature_cols = ['latitude', 'longitude', 'day_of_year']
    for col in ['temperature', 'salinity', 'depth']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
            feature_cols.append(col)

    # Interaction terms
    df['lat_lon_interaction'] = df['latitude'] * df['longitude']
    df['temp_salinity_interaction'] = df['temperature'] * df['salinity']
    feature_cols += ['lat_lon_interaction', 'temp_salinity_interaction']

    # Encode target variable
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['target_species'])

    # Get feature matrix
    X = df[feature_cols].values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"✅ Training set: {X_train.shape[0]} samples")
    print(f"✅ Test set: {X_test.shape[0]} samples")
    print(f"✅ Features: {feature_cols}")

    return X_train, X_test, y_train, y_test, label_encoder, feature_cols


def handle_imbalance(X_train, y_train):
    """
    Handle class imbalance using SMOTE oversampling.
    """
    print("⚠️ Addressing class imbalance with SMOTE...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"✅ Resampled training set: {X_resampled.shape[0]} samples")
    return X_resampled, y_resampled


def train_model(X_train, y_train, random_state=42):
    """
    Train a RandomForest classifier with class_weight='balanced'.
    """
    print("🚀 Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("✅ Model training complete")
    return model


def train_gradient_boosting_model(X_train, y_train, random_state=42):
    """
    Train a Gradient Boosting classifier.
    """
    print("🚀 Training Gradient Boosting model...")
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    print("✅ Model training complete")
    return model


def evaluate_model(model, X_test, y_test, label_encoder):
    """
    Evaluate the model and return metrics.
    """
    print("📊 Evaluating model...")
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    print(f"✅ Accuracy: {accuracy:.4f}")
    print(f"✅ Macro F1: {macro_f1:.4f}")

    # Classification report
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)

    # Confusion matrix
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, display_labels=label_encoder.classes_, xticks_rotation='vertical'
    )
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.close()
    print(f"✅ Confusion matrix saved to {CONFUSION_MATRIX_PATH}")

    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'class_metrics': report
    }


def save_model_and_metrics(model, metrics, label_encoder, feature_names):
    """
    Save the model, metrics, and metadata.
    """
    print("💾 Saving model and metrics...")
    model_data = {
        'model': model,
        'label_encoder': label_encoder,
        'feature_names': feature_names,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(MODEL_OUTPUT_PATH, 'wb') as f:
        pickle.dump(model_data, f)
    with open(METRICS_OUTPUT_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Model saved to {MODEL_OUTPUT_PATH}")
    print(f"✅ Metrics saved to {METRICS_OUTPUT_PATH}")


def main():
    """
    Main function to train and evaluate the model.
    """
    print("🌟 Species Classification Model Training 🌟")
    print("=========================================")

    # Load and prepare data
    X_train, X_test, y_train, y_test, label_encoder, feature_names = load_and_prepare_data(
        PROCESSED_DATA_PATH, top_n_species=20
    )

    # Handle class imbalance
    X_train, y_train = handle_imbalance(X_train, y_train)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test, label_encoder)

    # Save model and metrics
    save_model_and_metrics(model, metrics, label_encoder, feature_names)

    print("🎉 Training complete! 🎉")


if __name__ == "__main__":
    main()
