import json

def predict():
    # Another mock model
    return {"species": "salmon", "confidence": 0.85}

if __name__ == "__main__":
    print(json.dumps(predict()))
