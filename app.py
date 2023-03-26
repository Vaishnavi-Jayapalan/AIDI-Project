from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Efficientnetb0 imports
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
#from tensorflow.keras.applications.efficientnet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/EfficientnetB0.h5'

# Load your trained model

custom_objects = {'FixedDropout': tf.keras.layers.Dropout}  # Add the custom object
model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)

# Load the list of species names
species_path="class_info/name_of_spesies.txt"
with open(species_path, 'r') as f:
    species_list = [line.strip() for line in f]

print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):

    # Load the image
    # Preprocessing the image
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(256, 256))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    train_datagen= ImageDataGenerator(rescale=1.0/255)
    x = train_datagen.flow(x, shuffle=False).next()
    
    # Predicting the class
    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Get the predicted class name
        pred_class_idx = np.argmax(preds)
        pred_class = species_list[pred_class_idx]
        result = str(pred_class)
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)

