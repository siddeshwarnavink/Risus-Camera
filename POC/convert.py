import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the .hdf5 model
model = load_model('smile.hdf5')

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model to a .tflite file
with open('smile.tflite', 'wb') as f:
    f.write(tflite_model)