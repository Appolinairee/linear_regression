from flask import Flask, request, jsonify
import joblib
import numpy as np

model = joblib.load('model.joblib')

feature_names = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
    'PTRATIO', 'B', 'LSTAT', 'MEDV'
]

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if not all(feature in data for feature in feature_names):
            return jsonify({'error': 'Missing some required features'}), 400

        features = [data[feature] for feature in feature_names]
        
        features = np.array(features).reshape(1, -1)
        
        bias = np.ones((features.shape[0], 1))
        features_with_bias = np.concatenate([bias, features], axis=1)
        
        prediction = np.dot(model.T, features_with_bias.T)
        
        return jsonify({'prediction': prediction.tolist()})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
