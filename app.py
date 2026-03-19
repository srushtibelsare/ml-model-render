from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Model aur Scaler load karein
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return "Weather Prediction API is Working"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['input']
        # Data ko pehle scale karein
        scaled_data = scaler.transform([data])
        result = model.predict(scaled_data)
        return jsonify({'result': result.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

# Render ke liye 'app.run' ki zarurat nahi hoti, Gunicorn handles it.
