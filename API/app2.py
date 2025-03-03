import os
from flask import Flask, jsonify, request
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the model
model = pickle.load(open('animal_disease_classifier.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return 'Hello, World!'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        animal_name = request.form.get('animal_name')
        
        # Get symptoms (allow fewer than 5 symptoms)
        symptoms = []
        for i in range(1, 6):  # Check for symptoms1 to symptoms5
            symptom = request.form.get(f'symptoms{i}')
            if symptom:  # Only add if the symptom is provided
                symptoms.append(symptom)
        
        # Check if at least one symptom is provided
        if not animal_name or not symptoms:
            return jsonify({'error': 'Missing required input: animal_name and at least one symptom'}), 400
        
        # Combine symptoms into a single string
        symptoms_str = ','.join(symptoms)
        
        # Encode the inputs using LabelEncoder
        label_encoder = LabelEncoder()
        animal_name_encoded = label_encoder.fit_transform([animal_name])  # Encode animal name
        symptoms_encoded = label_encoder.fit_transform([symptoms_str])  # Encode symptoms
        
        # Prepare input query for the model
        input_query = np.array([animal_name_encoded[0], symptoms_encoded[0]]).reshape(1, -1)
        
        # Make a prediction
        result = model.predict(input_query)[0]
        
        # Return the result as JSON
        return jsonify({'The animal may be suffering from an animal disease.': str(result)})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)