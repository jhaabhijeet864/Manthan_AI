import json
import os
import sys
from datetime import datetime
import random

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

# Import the species model
from scripts.utils.species_model import predict as species_predict


def predict():
    """
    Generate predictions using trained models.
    Returns a JSON-serializable dictionary with predictions.
    """
    # Get species prediction from the trained model
    species_prediction = species_predict()
    
    # Enhance the prediction with additional oceanographic data
    # In a real system, this would come from real-time sources
    result = {
        "species": species_prediction["species"],
        "confidence": species_prediction["confidence"],
        "location": {
            "latitude": species_prediction["latitude"],
            "longitude": species_prediction["longitude"],
            "name": "Arabian Sea" if random.random() > 0.5 else "Bay of Bengal"
        },
        "oceanographic": {
            "temperature": round(24 + random.uniform(-2, 2), 1),
            "salinity": round(35 + random.uniform(-1, 1), 1),
            "oxygen": round(4 + random.uniform(-0.5, 0.5), 1)
        },
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return result


if __name__ == "__main__":
    print(json.dumps(predict()))
