import os
import sys
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

# Enhanced custom objects for better compatibility
custom_objects = {
    'RandomFlip': tf.keras.layers.experimental.preprocessing.RandomFlip,
    'RandomRotation': tf.keras.layers.experimental.preprocessing.RandomRotation,
    'RandomZoom': tf.keras.layers.experimental.preprocessing.RandomZoom,
    'RandomContrast': tf.keras.layers.experimental.preprocessing.RandomContrast,
    'RandomBrightness': tf.keras.layers.experimental.preprocessing.RandomBrightness,
    'EfficientNetB0': tf.keras.applications.EfficientNetB0
}

# Robust model loading with multiple fallbacks
def load_model_with_fallback():
    model_paths = [
        'model_assets/blackgram_model',  # Try SavedModel format first
        'model_assets/blackgram_model.keras'  # Fallback to .keras
    ]
    
    for path in model_paths:
        try:
            model = tf.keras.models.load_model(
                path,
                custom_objects=custom_objects,
                compile=False
            )
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            print(f"✅ Model loaded successfully from {path}")
            return model
        except Exception as e:
            print(f"⚠️ Failed to load from {path}: {str(e)}")
    
    return None

# Initialize model and class names
model = load_model_with_fallback()
class_names = []
if model:
    try:
        with open('model_assets/class_names.json') as f:
            class_names = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load class names: {str(e)}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Enhanced image preprocessing with validation"""
    try:
        img = Image.open(image_path).convert('RGB')
        if img.size != IMG_SIZE:
            img = img.resize(IMG_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
        return np.expand_dims(img_array, axis=0).astype('float32')
    except Exception as e:
        raise ValueError(f"Image processing failed: {str(e)}")

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({
            'error': 'Model not available',
            'status': 'service_unavailable'
        }), 503
        
    if 'file' not in request.files:
        return jsonify({
            'error': 'No file provided',
            'status': 'bad_request'
        }), 400
        
    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({
            'error': 'Empty file submission',
            'status': 'bad_request'
        }), 400
            
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
            'probabilities': {
                cls: float(pred) 
                for cls, pred in zip(class_names, predictions)
            },
            'status': 'success'
        })
    except ValueError as e:
        return jsonify({
            'error': f"Invalid image: {str(e)}",
            'status': 'unprocessable_entity'
        }), 422
    except Exception as e:
        return jsonify({
            'error': f"Prediction failed: {str(e)}",
            'status': 'internal_error'
        }), 500
    finally:
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception as e:
                print(f"Warning: Failed to remove temporary file {filepath}: {str(e)}")

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'operational' if model else 'degraded',
        'model_loaded': bool(model),
        'classes_loaded': len(class_names) > 0,
        'system': {
            'tensorflow_version': tf.__version__,
            'python_version': '.'.join(map(str, sys.version_info[:3])),
            'environment': 'production'
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)