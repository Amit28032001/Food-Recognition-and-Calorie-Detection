import tensorflow as tf
import h5py
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np

from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the saved model
model = tf.keras.models.load_model('best_model_3class.hdf5')

# Define the class labels
# Replace with your actual class labels
class_labels = ['samosa: 262 cal', 'pizza: 264 cal', 'omelette: 154 cal']


# Function to preprocess an image


def predict_class(model, images, show=True):
    for img in images:
        img = image.load_img(img, target_size=(299, 299))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img /= 255.

        pred = model.predict(img)
        print(pred)
        index = np.argmax(pred)
        class_labels.sort()

        pred_value = class_labels[index]
        print(pred_value)

        if show:
            plt.imshow(img[0])
            plt.axis('off')
            plt.title(pred_value)
            plt.show()


# Example usage
image_path = [r'C:\Users\sanja\Desktop\Food API\pizza.jpg']  # Replace with the path to your image
predict_class(model, image_path, True)
# print('Confidence:', confidence)
