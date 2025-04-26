from flask import Flask, jsonify, request
from flask_cors import CORS
from database import get_connection
import uuid

app = Flask(__name__)
CORS(app)

@app.route('/api/preprocess', methods=['POST'])
def prep():
    data = request.get_json()
    
    if 'features' not in data:
        return jsonify({'error': 'Missing features in request'}), 400
    
    features = np.array(data['features']).reshape(1, -1)
    return jsonify(features), 201

@app.route('/api/predict', methods=['POST'])    
def create_prediction():
    prediction = model.predict(features).tolist()
    
    prediction_id = str(uuid.uuid4())
    prediction_results[prediction_id] = prediction
    response = {
        'id' : prediction_id,
        'prediction' : prediction
    }
    return jsonify(response), 201


if __name__ == '__main__':
    app.run(debug=True)