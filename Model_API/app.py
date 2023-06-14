import flask
import io
import string
import time
import os
import numpy as np
import tensorflow as tf
from tensorflow.nn import softmax
from numpy import argmax
from PIL import Image
from flask import Flask, jsonify, request 

model = tf.keras.models.load_model('pelita_mobilenetv2.h5')

def prepare_image(img):
    img = Image.open(io.BytesIO(img))
    img = img.resize((300, 300))
    img = np.array(img)
    img = np.expand_dims(img, 0)
    return img

def predict_result(img):
    labels = ['battery', 'cable', 'cups', 'kettle', 'keyboard', 'lamp', 'laptop', 'mineral_bottle', 'monitor', 'mouse', 'phone', 'plastic_bag', 'rice_cooker', 'spray_bottle', 'toothbrush']
    pred = model.predict(img)
    score = softmax(pred[0])
    
    return labels[argmax(score)]

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def infer_image():
    # Catch the image file from a POST request
    if 'file' not in request.files:
        return "Please try again. The Image doesn't exist"

    file = request.files.get('file')

    if not file:
        return

    # Read the image
    img_bytes = file.read()

    # Prepare the image
    img = prepare_image(img_bytes)

    # Return on a JSON format
    return jsonify(prediction=predict_result(img))

@app.route('/', methods=['GET'])
def index():
    return 'Machine Learning Inference'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')