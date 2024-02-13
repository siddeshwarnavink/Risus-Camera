import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
from PIL import Image, ImageTk

interpreter = tf.lite.Interpreter(model_path="smile.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def process_frame(frame):
    resized_image = cv2.resize(frame, (32, 32))
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    normalized_image = gray_image / 255.0

    input_data = np.expand_dims(normalized_image, axis=0)
    input_data = np.expand_dims(input_data, axis=-1).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    if output_data[0] >= 0.5:
        return "Smile detected", (0, 255, 0)
    else:
        return "No smile detected", (0, 0, 255)

def webcam_feed(root):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to open webcam.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Unable to capture frame.")
            break

        processed_text, text_color = process_frame(frame)
        cv2.putText(frame, processed_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

        cv2.imshow("Smile detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    root.destroy()

root = tk.Tk()
root.title("Smile detection")

webcam_feed(root)
root.mainloop()
