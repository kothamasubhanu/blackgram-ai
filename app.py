import os
import tensorflow as tf
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import json
import numpy as np
from PIL import Image
import gdown  # For backup weight downloads

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'tmp_uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
IMG_SIZE = (128, 128)
MODEL_URL = "https://drive.google.com/uc?id=YOUR_MODEL_FILE_ID"  # Replace with your actual file ID

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB').resize(IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    return tf.expand_dims(img_array, 0)

def load_model_with_fallbacks():
    """Robust model loading with multiple fallback mechanisms"""
    model_paths = [
        'blackgram_model',               # 1. Try SavedModel format first
        'blackgram_model.keras',         # 2. Try native Keras format
        'model_weights.h5'               # 3. Try weights only
    ]
    
    # Attempt 1: Load from any available format
    for path in model_paths:
        try:
            if path.endswith('.h5'):
                model = build_model_from_scratch()
                model.load_weights(path)
                print(f"✅ Loaded weights from {path}")
            else:
                model = tf.keras.models.load_model(path)
                print(f"✅ Loaded full model from {path}")
            return model
        except Exception as e:
            print(f"⚠️ Failed loading {path}: {str(e)}")
    
    # Attempt 2: Download weights if all else fails
    try:
        if not os.path.exists('model_weights.h5'):
            print("⬇️ Downloading weights from Google Drive...")
            gdown.download(MODEL_URL, 'model_weights.h5', quiet=False)
        model = build_model_from_scratch()
        model.load_weights('model_weights.h5')
        print("✅ Model loaded from downloaded weights")
        return model
    except Exception as e:
        print(f"❌ All loading methods failed: {str(e)}")
        raise

def build_model_from_scratch():
    """Rebuild model architecture from scratch"""
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        input_shape=(128, 128, 3),
        weights='imagenet'
    )
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(5, activation='softmax')(x)
    return tf.keras.Model(inputs=base_model.input, outputs=outputs)

# Initialize model and classes
try:
    model = load_model_with_fallbacks()
    with open('class_names.json') as f:
        class_names = json.load(f)
    print(f"🚀 Model ready. Classes: {class_names}")
except Exception as e:
    print(f"🔥 Critical initialization error: {str(e)}")
    model = None
    class_names = []

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded', 'status': 'failed'}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded', 'status': 'failed'}), 400
    
    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({'error': 'No selected file', 'status': 'failed'}), 400
        
    if not allowed_file(file.filename):
        return jsonify({
            'error': 'Allowed file types: png, jpg, jpeg',
            'status': 'failed'
        }), 400

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        img_array = preprocess_image(filepath)
        predictions = model.predict(img_array, verbose=0)[0]
        confidence = float(np.max(predictions))
        
        return jsonify({
            "prediction": class_names[np.argmax(predictions)],
            "confidence": round(confidence, 4),
            "all_predictions": dict(zip(class_names, predictions.tolist())),
            "status": "success"
        })
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'failed'}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

@app.route('/')
def health_check():
    return jsonify({
        "status": "ready" if model else "error",
        "model": "Blackgram Disease Detector",
        "classes": class_names,
        "model_type": "EfficientNetB0",
        "input_size": IMG_SIZE
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)