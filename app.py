import os
import tensorflow as tf
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import json
import numpy as np
from PIL import Image

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'tmp_uploads'  # Changed to tmp_uploads for clarity
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
IMG_SIZE = (128, 128)  # Must match your training size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB limit

# Ensure upload folder exists and has proper permissions
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.chmod(UPLOAD_FOLDER, 0o755)  # Read/write permissions

# Load model and class names
try:
    model = tf.keras.models.load_model('blackgram_model.keras')
    with open('class_names.json') as f:
        class_names = json.load(f)
    print("✅ Model loaded successfully. Classes:", class_names)
except Exception as e:
    print("❌ Model loading failed:", str(e))
    class_names = []
    model = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB').resize(IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch dimension
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded', 'status': 'failed'}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded', 'status': 'failed'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file', 'status': 'failed'}), 400
        
    if not allowed_file(file.filename):
        return jsonify({
            'error': 'Allowed file types: png, jpg, jpeg',
            'status': 'failed'
        }), 400

    try:
        # Secure save temporary file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process and predict
        img_array = preprocess_image(filepath)
        predictions = model.predict(img_array, verbose=0)[0]
        confidence = float(np.max(predictions))
        predicted_class = class_names[np.argmax(predictions)]
        
        return jsonify({
            "prediction": predicted_class,
            "confidence": round(confidence, 4),
            "all_predictions": {
                cls: float(pred) for cls, pred in zip(class_names, predictions)
            },
            "status": "success"
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'failed'
        }), 500
        
    finally:
        # Always clean up
        if os.path.exists(filepath):
            os.remove(filepath)

@app.route('/')
def health_check():
    return jsonify({
        "status": "ready" if model else "loading_failed",
        "model": "Blackgram Disease Detector",
        "classes": class_names,
        "upload_folder": os.path.abspath(app.config['UPLOAD_FOLDER'])
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
def load_model_with_retry():
    try:
        model = tf.keras.models.load_model('blackgram_model.keras')
        print("✅ Model loaded directly")
        return model
    except:
        print("⚠️ Attempting backup load method...")
        try:
            # Reconstruct from layers (works without file)
            base_model = tf.keras.applications.EfficientNetB0(
                include_top=False, 
                input_shape=(128,128,3)
            )
            x = base_model.output
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Dropout(0.5)(x)
            x = tf.keras.layers.Dense(256, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.3)(x)
            outputs = tf.keras.layers.Dense(5, activation='softmax')(x)
            model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
            model.load_weights('blackgram_model.keras')  # Load weights only
            print("✅ Model reconstructed from architecture")
            return model
        except Exception as e:
            print(f"❌ Critical error: {str(e)}")
            raise

model = load_model_with_retry()