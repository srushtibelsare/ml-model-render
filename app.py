from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return "Working"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['input']
    result = model.predict([data])
    return jsonify({'result': result.tolist()})

app.run()