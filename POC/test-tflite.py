import h5py
import cv2
import numpy as np
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="smile.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

#image_path = 'sample_images/no-smile1.jpg'
image_path = 'sample_images/smile1.jpg'
image = cv2.imread(image_path)

# Preprocess the image
resized_image = cv2.resize(image, (32, 32))
gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

# Normalize the image
normalized_image = gray_image / 255.0

input_data = np.expand_dims(normalized_image, axis=0)
input_data = np.expand_dims(input_data, axis=-1).astype(np.float32)

interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])

if output_data[0] >= 0.5:
    print("Smile detected. Take a photo!")
else:
    print("No smile detected.")