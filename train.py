import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import json
import os
import hashlib
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Configuration for reproducibility
tf.keras.backend.set_floatx('float32')
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = '42'
np.random.seed(42)
tf.random.set_seed(42)

# Constants
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

def load_and_prepare_data():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        r"E:\blackgram_dataset\train_balanced",
        label_mode="categorical",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        r"E:\blackgram_dataset\valid",
        label_mode="categorical",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42
    )
    return train_ds, val_ds

def build_model(class_names):
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.25),
        layers.RandomZoom(0.25),
        layers.RandomContrast(0.25),
        layers.RandomBrightness(0.2),
    ])

    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(len(class_names), activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model, base_model

def train_model(model, train_ds, val_ds):
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
        tf.keras.callbacks.ModelCheckpoint(
            "best_model.keras",
            save_best_only=True,
            monitor='val_accuracy',
            mode='max'
        )
    ]
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30,
        callbacks=callbacks,
        verbose=2
    )
    return history

def fine_tune_model(model, base_model, train_ds, val_ds):
    base_model.trainable = True
    for layer in base_model.layers[:-60]:
        layer.trainable = False
        
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        verbose=2
    )
    return history

def save_model(model, class_names):
    """Simplified saving that works with Keras 3"""
    os.makedirs('model_assets', exist_ok=True)
    
    # Save model in Keras format
    model_path = "model_assets/blackgram_model.keras"
    model.save(model_path)
    print(f"✅ Model saved to {model_path}")
    
    # Save weights separately as backup
    weights_path = "model_assets/model.weights.h5"
    model.save_weights(weights_path)
    
    # Save class names
    with open("model_assets/class_names.json", "w") as f:
        json.dump(class_names, f)
    
    print(f"💾 Backup weights saved to {weights_path}")

def evaluate_model(model, val_ds, class_names):
    y_true = []
    y_pred = []
    
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(np.argmax(labels.numpy(), axis=1))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)
    with open('metrics.json', 'w') as f:
        json.dump(classification_report(y_true, y_pred, target_names=class_names, output_dict=True), f)

def main():
    tf.config.set_visible_devices([], 'GPU')  # Force CPU for consistency
    
    train_ds, val_ds = load_and_prepare_data()
    class_names = train_ds.class_names
    print("Class names:", class_names)
    
    model, base_model = build_model(class_names)
    
    print("\n=== Initial Training ===")
    train_model(model, train_ds, val_ds)
    
    print("\n=== Fine-Tuning ===")
    fine_tune_model(model, base_model, train_ds, val_ds)
    
    print("\n=== Saving Model ===")
    save_model(model, class_names)
    
    print("\n=== Evaluation ===")
    evaluate_model(model, val_ds, class_names)

if __name__ == "__main__":
    main()