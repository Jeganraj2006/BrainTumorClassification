import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import VGG16

# -------------------------------
# BASIC SETUP
# -------------------------------
IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS_STAGE1 = 15
EPOCHS_STAGE2 = 20

TRAIN_DIR = "dataset/train"
VAL_DIR   = "dataset/val"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices('GPU'))

NUM_CLASSES = len(os.listdir(TRAIN_DIR))
print("Classes:", os.listdir(TRAIN_DIR))

# -------------------------------
# DATA GENERATORS
# -------------------------------
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE, class_mode="categorical"
)

val_data = val_gen.flow_from_directory(
    VAL_DIR, target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False
)

# -------------------------------
# CALLBACKS
# -------------------------------
callbacks = [
    ModelCheckpoint("models/best_vgg.keras", monitor="val_accuracy",
                    save_best_only=True, verbose=1),
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.3, verbose=1)
]

# -------------------------------
# BUILD VGG16
# -------------------------------
base = VGG16(weights="imagenet", include_top=False,
            input_shape=(IMG_SIZE, IMG_SIZE, 3))
base.trainable = False

x = base.output
x = layers.Flatten()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
out = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = models.Model(base.input, out)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------------
# STAGE 1
# -------------------------------
print("\n--- STAGE 1: Training top layers ---")
model.fit(train_data, validation_data=val_data,
epochs=EPOCHS_STAGE1, callbacks=callbacks)

# -------------------------------
# STAGE 2 (Fine-tuning)
# -------------------------------
print("\n--- STAGE 2: Fine-tuning VGG16 ---")
base.trainable = True
for layer in base.layers[:-6]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(train_data, validation_data=val_data,
epochs=EPOCHS_STAGE2, callbacks=callbacks)

model.save("models/VGG16_Final.keras")
print("✅ VGG16 training finished")
