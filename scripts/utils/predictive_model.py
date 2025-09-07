import json

def predict():
    # Dummy prediction
    return {"species": "tuna", "confidence": 0.92}

if __name__ == "__main__":
    print(json.dumps(predict()))
