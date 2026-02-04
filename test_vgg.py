# import tensorflow as tf
# import numpy as np
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.vgg16 import preprocess_input

# # -------------------------
# # SETTINGS
# # -------------------------
# MODEL_PATH = "best_vgg16.keras"     # change if name is different
# IMG_PATH = "test2.jpg"         # put your MRI image here
# IMG_SIZE = 224

# # -------------------------
# # Load model
# # -------------------------
# print("✅ Loading VGG16 model...")
# model = tf.keras.models.load_model(MODEL_PATH)

# # -------------------------
# # Class labels (MUST match training)
# # -------------------------
# class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
# # If different, copy from: train_generator.class_indices

# # -------------------------
# # Load and preprocess image
# # -------------------------
# img = image.load_img(IMG_PATH, target_size=(IMG_SIZE, IMG_SIZE))
# img_array = image.img_to_array(img)
# img_array = np.expand_dims(img_array, axis=0)
# img_array = preprocess_input(img_array)   # ⚠ VERY IMPORTANT

# # -------------------------
# # Predict
# # -------------------------
# pred = model.predict(img_array)
# predicted_class = class_names[np.argmax(pred)]
# confidence = np.max(pred) * 100

# # -------------------------
# # Output
# # -------------------------
# print("\n🧠 VGG16 BRAIN TUMOR PREDICTION")
# print("Predicted Class :", predicted_class)
# print("Confidence     :", round(confidence, 2), "%")

# # Optional: show all probabilities
# print("\nClass probabilities:")
# for i in range(len(class_names)):
#     print(class_names[i], ":", round(pred[0][i]*100, 2), "%")




import tensorflow as tf

# PATHS
MODEL_PATH = "models/best_vgg.keras"
VAL_DIR = "dataset/val"
IMG_SIZE = 224
BATCH_SIZE = 32

# Load model
print("✅ Loading VGG16 model...")
model = tf.keras.models.load_model(MODEL_PATH)

# VGG preprocessing (VERY IMPORTANT)
from tensorflow.keras.applications.vgg16 import preprocess_input

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=preprocess_input
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False   # ⚠ MUST be False
)

# Evaluate
loss, acc = model.evaluate(val_generator)

print("\n🎯 VGG16 VALIDATION RESULT")
print("Loss     :", loss)
print("Accuracy :", acc * 100, "%")

print("\nClass order:", val_generator.class_indices)
