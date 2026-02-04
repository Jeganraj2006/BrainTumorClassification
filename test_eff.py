# import tensorflow as tf
# import numpy as np
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.efficientnet import preprocess_input

# # Paths
# MODEL_PATH = "best_effnet.keras"
# IMAGE_PATH = "dataset/train/no_tumor/0056.jpg"
# IMG_SIZE = 224

# # Class labels (must match training)
# class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']


# # Load model
# print("✅ Loading EfficientNet model...")
# model = tf.keras.models.load_model(MODEL_PATH)

# # Load & preprocess image
# img = image.load_img(IMAGE_PATH, target_size=(IMG_SIZE, IMG_SIZE))
# img_array = image.img_to_array(img)
# img_array = np.expand_dims(img_array, axis=0)
# img_array = preprocess_input(img_array)  
# # Predict
# predictions = model.predict(img_array)[0]

# predicted_class = class_names[np.argmax(predictions)]
# confidence = np.max(predictions) * 100

# # Show all class probabilities
# print("\n📊 Class Probabilities:")
# for i, prob in enumerate(predictions):
#     print(f"{class_names[i]:12s}: {prob*100:.2f}%")

# print("\n🎯 FINAL PREDICTION")
# print("Tumor Type :", predicted_class)
# print("Confidence :", f"{confidence:.2f}%")




import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input

VAL_DIR = "dataset/val"
IMG_SIZE = 224
BATCH_SIZE = 32

model = tf.keras.models.load_model("models/best_effnet.keras")
print("✅ Best EfficientNet model loaded")

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

val_data = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

loss, accuracy = model.evaluate(val_data)

print("\n🎯 EfficientNet Validation Results")
print("Validation Loss     :", loss)
print("Validation Accuracy :", accuracy * 100, "%")

print("\nClass order:", val_data.class_indices)
