import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import json
import os
from sklearn.utils.class_weight import compute_class_weight

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# 1. Load datasets
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

class_names = train_ds.class_names
print("Class names:", class_names)

# Save class names for inference
with open("class_names.json", "w") as f:
    json.dump(class_names, f)

# 2. Data augmentation and preprocessing
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.25),
    layers.RandomZoom(0.25),
    layers.RandomContrast(0.25),
    layers.RandomBrightness(0.2),
])

from tensorflow.keras.applications.efficientnet import preprocess_input

# 3. Build the model (Functional API)
inputs = layers.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = preprocess_input(x)
base_model = tf.keras.applications.EfficientNetB0(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze base for initial training
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.5)(x)  # Stronger dropout for regularization
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(class_names), activation='softmax')(x)
model = models.Model(inputs, outputs)

# 4. Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 5. Callbacks
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

callbacks = [
    EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
    ModelCheckpoint("best_model_efficientnet.keras", save_best_only=True)
]

# 6. Compute class weights (optional, if classes are imbalanced)
# If all classes have 500 images, you can skip this.
# If not, uncomment below:
# y_train = []
# for _, labels in train_ds.unbatch():
#     y_train.append(np.argmax(labels.numpy()))
# class_weights = compute_class_weight('balanced', classes=np.arange(len(class_names)), y=y_train)
# class_weights = dict(enumerate(class_weights))
class_weights = None

# 7. Initial training
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    callbacks=callbacks,
    class_weight=class_weights
)

# 8. Fine-tune: Unfreeze last 60 layers of EfficientNetB0
base_model.trainable = True
for layer in base_model.layers[:-60]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),  # Lower LR for fine-tuning
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history_finetune = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=callbacks,
    class_weight=class_weights
)

# 9. Save final model
model.save("trained_blackgram_model_efficientnet.keras")
print("Model and class names saved.")

# 10. Evaluate with confusion matrix and classification report
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

y_true = []
y_pred = []

for images, labels in val_ds:
    preds = model.predict(images)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(np.argmax(labels.numpy(), axis=1))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

print(classification_report(y_true, y_pred, target_names=class_names))

# [Your exact model training code here]
# Add these lines at the very end:
if __name__ == "__main__":
    # Force CPU usage for consistent behavior
    tf.config.set_visible_devices([], 'GPU')
    
    # Run training and save final model
    print("Starting training process...")
    model.save('trained_blackgram_model_efficientnet.keras')
    
    # Verify model can be loaded
    try:
        tf.keras.models.load_model('trained_blackgram_model_efficientnet.keras')
        print("✅ Model saved and verified successfully")
    except Exception as e:
        print("❌ Model verification failed:", str(e))