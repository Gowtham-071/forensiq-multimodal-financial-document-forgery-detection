"""
scripts/train_cnn.py
CNN Retraining on balanced dataset/ — run from project root
Usage: python scripts/train_cnn.py
"""

import os, sys, time
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# ── Tensorflow (suppress info logs) ──────────────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

TRAIN_DIR  = BASE_DIR / 'dataset' / 'train'
VAL_DIR    = BASE_DIR / 'dataset' / 'val'
MODEL_PATH = str(BASE_DIR / 'models' / 'fraud_document_cnn.h5')
IMG_SIZE   = 128
BATCH      = 32
EPOCHS     = 30

# ── Banner ────────────────────────────────────────────────────────────────────
print()
print("╔══════════════════════════════════════════════════════╗")
print("║       FORENSIQ — CNN Retraining on dataset/          ║")
print("╚══════════════════════════════════════════════════════╝")
print(f"  TensorFlow : {tf.__version__}")
print(f"  GPU        : {len(tf.config.list_physical_devices('GPU')) > 0}")
print(f"  Train dir  : {TRAIN_DIR}")

n_train_f = len(list((TRAIN_DIR/'fraud').iterdir()))
n_train_g = len(list((TRAIN_DIR/'genuine').iterdir()))
n_val_f   = len(list((VAL_DIR/'fraud').iterdir()))
n_val_g   = len(list((VAL_DIR/'genuine').iterdir()))

print(f"\n  Train: {n_train_f} fraud + {n_train_g} genuine = {n_train_f+n_train_g}")
print(f"  Val  : {n_val_f} fraud + {n_val_g} genuine = {n_val_f+n_val_g}")
print()

# ── Data generators ───────────────────────────────────────────────────────────
train_gen = ImageDataGenerator(
    rescale=1./255, rotation_range=5,
    width_shift_range=0.05, height_shift_range=0.05,
    zoom_range=0.05, fill_mode='nearest'
).flow_from_directory(
    str(TRAIN_DIR), target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale', batch_size=BATCH,
    class_mode='binary', classes=['fraud', 'genuine'],
    shuffle=True, seed=42
)

val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    str(VAL_DIR), target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale', batch_size=BATCH,
    class_mode='binary', classes=['fraud', 'genuine'], shuffle=False
)

print(f"\n  Class indices: {train_gen.class_indices}")

# ── Model ─────────────────────────────────────────────────────────────────────
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', padding='same',
                  input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    layers.BatchNormalization(), layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(), layers.MaxPooling2D(2,2), layers.Dropout(0.25),

    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(), layers.MaxPooling2D(2,2), layers.Dropout(0.25),

    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(), layers.GlobalAveragePooling2D(),

    layers.Dense(128, activation='relu'), layers.Dropout(0.4),
    layers.Dense(1, activation='sigmoid')
])

total_params = model.count_params()
print(f"\n  Model params: {total_params:,}")

# Class weights
total = n_train_f + n_train_g
class_weights = {0: total/(2*n_train_f), 1: total/(2*n_train_g)}
print(f"  Class weights: fraud={class_weights[0]:.2f}, genuine={class_weights[1]:.2f}")

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy',
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall')]
)

# Callbacks
cbs = [
    callbacks.ModelCheckpoint(MODEL_PATH, monitor='val_accuracy',
                               save_best_only=True, verbose=0),
    callbacks.EarlyStopping(monitor='val_accuracy', patience=6,
                            restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                patience=3, verbose=1, min_lr=1e-6),
]

print(f"\n  Training for up to {EPOCHS} epochs (EarlyStopping patience=6)")
print("  Best model → models/fraud_document_cnn.h5")
print("  ─────────────────────────────────────────────────────")
print()

# ── Train ─────────────────────────────────────────────────────────────────────
t0 = time.time()

history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    class_weight=class_weights,
    callbacks=cbs,
    verbose=1   # Keras built-in progress bars per epoch
)

elapsed = round((time.time() - t0) / 60, 1)

# ── Results ───────────────────────────────────────────────────────────────────
best_val_acc = max(history.history['val_accuracy'])
best_epoch   = history.history['val_accuracy'].index(best_val_acc) + 1
epochs_run   = len(history.history['val_accuracy'])

print()
print("╔══════════════════════════════════════════════════════╗")
print("║                  TRAINING COMPLETE                    ║")
print("╚══════════════════════════════════════════════════════╝")
print(f"  Epochs run     : {epochs_run}/{EPOCHS}")
print(f"  Best val acc   : {best_val_acc*100:.1f}% (epoch {best_epoch})")
print(f"  Target         : 85%  → {'✅ PASSED' if best_val_acc >= 0.85 else '⚠️ BELOW TARGET'}")
print(f"  Time elapsed   : {elapsed} min")
print(f"  Model saved to : {MODEL_PATH}")
print()

# Save training curves
try:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    reports = BASE_DIR / 'reports'
    reports.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(history.history['accuracy'],     label='Train', color='#58a6ff')
    ax[0].plot(history.history['val_accuracy'], label='Val',   color='#3fb950')
    ax[0].axhline(0.85, color='#f85149', linestyle='--', label='85% target')
    ax[0].set_title('Accuracy'); ax[0].legend(); ax[0].set_ylim([0,1])
    ax[1].plot(history.history['loss'],     label='Train', color='#58a6ff')
    ax[1].plot(history.history['val_loss'], label='Val',   color='#3fb950')
    ax[1].set_title('Loss'); ax[1].legend()
    plt.tight_layout()
    plt.savefig(str(reports / 'training_curves.png'), dpi=150)
    print("  Training curves saved → reports/training_curves.png")
except Exception as e:
    print(f"  (Could not save curves: {e})")

print("\n▶ Next: python scripts/run_evaluation.py")
