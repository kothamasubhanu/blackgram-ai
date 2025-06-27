import os
import tensorflow as tf
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import json

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
IMG_SIZE = (128, 128)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Custom objects for model loading
custom_objects = {
    'RandomFlip': tf.keras.layers.RandomFlip,
    'RandomRotation': tf.keras.layers.RandomRotation,
    'RandomZoom': tf.keras.layers.RandomZoom,
    'RandomContrast': tf.keras.layers.RandomContrast,
    'RandomBrightness': tf.keras.layers.RandomBrightness
}

# Load model with error handling
try:
    model = tf.keras.models.load_model(
        'model_assets/blackgram_model.keras',
        custom_objects=custom_objects,
        compile=False
    )
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    with open('model_assets/class_names.json') as f:
        class_names = json.load(f)
        
    print("✅ Model and class names loaded successfully")
except Exception as e:
    print(f"❌ Failed to load model: {str(e)}")
    model = None
    class_names = []

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Preprocess image for model prediction"""
    try:
        img = Image.open(image_path).convert('RGB').resize(IMG_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        raise ValueError(f"Image processing failed: {str(e)}")

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded', 'status': 'service_unavailable'}), 503
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided', 'status': 'bad_request'}), 400
        
    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({'error': 'Empty file submission', 'status': 'bad_request'}), 400
            
    if not allowed_file(file.filename):
        return jsonify({
            'error': 'Invalid file type',
            'allowed_types': list(ALLOWED_EXTENSIONS),
            'status': 'unsupported_media_type'
        }), 415

    filepath = None
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        img_tensor = preprocess_image(filepath)
        predictions = model.predict(img_tensor, verbose=0)[0]
        
        return jsonify({
            'prediction': class_names[np.argmax(predictions)],
            'confidence': float(np.max(predictions)),
            'probabilities': {cls: float(pred) for cls, pred in zip(class_names, predictions)},
            'status': 'success'
        })
    except ValueError as e:
        return jsonify({'error': str(e), 'status': 'unprocessable_entity'}), 422
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'internal_error'}), 500
    finally:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'operational' if model else 'degraded',
        'model_loaded': bool(model),
        'classes_loaded': len(class_names) if model else 0,
        'system': {
            'tensorflow_version': tf.__version__,
            'python_version': '.'.join(map(str, sys.version_info[:3]))
        }
    })

if __name__ == '__main__':
    import sys
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)