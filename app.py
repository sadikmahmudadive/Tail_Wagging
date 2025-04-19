import os
from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)

# Config
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
IMG_SIZE = 224  # Match this with your model
MODEL_PATH = 'pet_disease_model.tflite'
CLASS_NAMES_PATH = 'class_names.txt'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load class names (removing "|Unknown|Not Available")
with open(CLASS_NAMES_PATH, 'r') as f:
    class_names = [line.strip() for line in f if line.strip() != "|Unknown|Not Available"]

# Optional: add medicine recommendations
disease_medicine_map = {
    "Dental Disease in Cat": ["Chlorhexidine oral rinse", "Virbac C.E.T toothpaste", "PlaqueOff powder"],
    "Dental Disease in Dog": ["Vet Enzymatic Toothpaste", "Chlorhexidine oral gel", "Clindamycin (Antirobe)"],
    "Distemper in Dog": ["Supportive care: IV fluids", "Vitamin B-complex", "Clavamox or Enrofloxacin for secondary infections"],
    "Ear Mites in Cat": ["Ivermectin drops", "Selamectin (Revolution)", "Otodex Ear Drops"],
    "Eye Infection in Cat": ["Tobramycin Eye Drops", "Tetracycline Eye Ointment", "Saline rinse"],
    "Eye Infection in Dog": ["Chloramphenicol Eye Drops", "Ciprofloxacin Eye Drops", "Tobramycin Eye Drops"],
    "Feline Leukemia": ["Interferon-alpha", "Supportive therapy", "Broad-spectrum antibiotics for secondary infections"],
    "Feline Panleukopenia": ["IV fluids", "Ceftriaxone or Ampicillin", "Vitamin B complex"],
    "Fungal Infection in Cat": ["Ketoconazole cream", "Itraconazole", "Lime sulfur dip"],
    "Fungal Infection in Dog": ["Griseofulvin", "Miconazole shampoo", "Ketoconazole tablets"],
    "Hot Spots in Dog": ["Betadine solution", "Hydrocortisone cream", "Cephalexin or Amoxicillin"],
    "Kennel Cough in Dog": ["Doxycycline", "Cough suppressants like Temaril-P", "Steam therapy"],
    "Mange in Dog": ["Ivermectin", "Amitraz dip", "Selamectin (Revolution)"],
    "Parvovirus in Dog": ["IV fluids", "Metronidazole or Ceftriaxone", "Ondansetron (anti-vomiting)"],
    "Ringworm in Cat": ["Clotrimazole cream", "Griseofulvin", "Ketoconazole shampoo"],
    "Scabies in Cat": ["Selamectin", "Ivermectin", "Benzyl benzoate lotion"],
    "Skin Allergy in Cat": ["Antihistamines (Cetirizine)", "Omega 3 fatty acids", "Hydrocortisone cream"],
    "Skin Allergy in Dog": ["Loratadine", "Coconut oil rub", "Apoquel (if available)"],
    "Tick Infestation in Dog": ["Fipronil spray", "Ivermectin", "Bravecto (if accessible)"],
    "Urinary Tract Infection in Cat": ["Amoxicillin-clavulanic acid", "Cranberry extract", "Doxycycline"],
    "Worm Infection in Cat": ["Albendazole", "Pyrantel Pamoate", "Fenbendazole"],
    "Worm Infection in Dog": ["Praziquantel", "Albendazole", "Pyrantel Pamoate"]
}

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image_array = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)

def predict(image_array):
    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]

@app.route('/')
def home():
    return '🐾 Pet Disease Detection API - POST to /predict with a pet image.'

@app.route('/predict', methods=['POST'])
def predict_disease():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image']
    if not allowed_file(image.filename):
        return jsonify({'error': 'Allowed file types: png, jpg, jpeg'}), 400

    try:
        img = Image.open(io.BytesIO(image.read()))
        processed = preprocess_image(img)

        expected_shape = tuple(input_details[0]['shape'])
        if processed.shape != expected_shape:
            return jsonify({'error': f'Image shape mismatch. Expected {expected_shape}, got {processed.shape}'}), 400

        prediction = predict(processed)
        predicted_index = int(np.argmax(prediction))
        confidence = float(prediction[predicted_index])
        predicted_disease = class_names[predicted_index]

        # Get medicine recommendation (if available)
        medicines = disease_medicine_map.get(predicted_disease, ["No specific medicine info available"])

        result = {
            'predicted_disease': predicted_disease,
            'confidence': round(confidence, 4),
            'recommended_medicines': medicines
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
