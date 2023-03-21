from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import tensorflow as tf

import numpy as np

# Define the custom layer
class FixedDropout(tf.keras.layers.Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape

        symbolic_shape = tf.keras.backend.shape(inputs)
        noise_shape = [symbolic_shape[axis] if shape is None else shape
                       for axis, shape in enumerate(self.noise_shape)]
        return tuple(noise_shape)

# Register the custom layer with Keras
custom_objects = {'FixedDropout': FixedDropout}

# Load the model using the custom objects
model = load_model('models/EfficientnetB0.h5', custom_objects=custom_objects)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    img = request.files['image']
    img = image.load_img(img, target_size=(256, 256))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    pred_class = preds.argmax(axis=-1)
    return render_template('index.html', prediction=pred_class[0])

if __name__ == '__main__':
    app.run(debug=True)
