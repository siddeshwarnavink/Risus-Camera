import h5py
import cv2
import numpy as np
from keras.models import model_from_json

model_path = 'smile.hdf5'
with h5py.File(model_path, 'r') as f:
    model_json = f.attrs['model_config']
    loaded_model = model_from_json(model_json)
    loaded_model.load_weights(model_path)

#image_path = 'sample_images/no-smile1.jpg'
image_path = 'sample_images/smile1.jpg'
image = cv2.imread(image_path)

# Preprocess the image
resized_image = cv2.resize(image, (32, 32))
gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

# Perform prediction
input_data = np.expand_dims(gray_image, axis=0)
input_data = np.expand_dims(input_data, axis=-1)

prediction = loaded_model.predict(input_data)

# Interpret the prediction
if prediction[0] >= 0.5:
	print("Smile detected. Take a photo!")
else:
	print("No smile detected.")
