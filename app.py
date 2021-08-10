import pickle
from flask import Flask, render_template, request
import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model

app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/prediction', methods=['POST'])
def prediction():
    # model = load_model(r'venv/mask.h5')
    # img = request.files['img']
    # img.save('img.jpg')
    # image = cv2.imread('img.jpg')
    # image = image/255
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = cv2.resize(image, (224, 244))
    # image = np.reshape(image, (1, 224, 224, 3))
    # pred = model.predict(image)
    # pred = np.argmax(pred)
    model = load_model(r'venv/mask.h5')
    img = request.files['img']
    img.save('img.jpg')
    frame = cv2.imread('img.jpg')
    final_image = cv2.resize(frame, (224, 224))
    final_image = np.expand_dims(final_image, axis=0)
    final_image = final_image / 255.0
    pred = model.predict(final_image)
    return render_template('prediction.html', data=pred)

print(tf.__version__)
if __name__ == '__main__':
    app.run(port=3000, debug=True)
