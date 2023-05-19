from flask import Flask, jsonify, render_template, request, send_file

from bson import ObjectId
from PIL import Image
import io
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)


# load TensorFlow model for clothing image recognition
model_path = "model/model_20k.h5"
model = tf.keras.models.load_model(model_path)
model.summary()


@app.route("/")
def hello():
    return "hello world"


# API route for uploading an image and getting recommendations
@app.route("/generate_image", methods=["POST"])
def generate_image():
    # retrieve image data from request

    img1 = Image.open(io.BytesIO(request.files["image1"].read()))
    img2 = Image.open(io.BytesIO(request.files["image2"].read()))

    # resize images to 32x32
    img1 = img1.resize((32, 32))
    img2 = img2.resize((32, 32))

    # convert images to arrays and normalize
    img1_array = np.array(img1) / 255.0
    img2_array = np.array(img2) / 255.0

    # expand dimensions of arrays to match model input shape
    img1_tensor = tf.expand_dims(img1_array, 0)
    img2_tensor = tf.expand_dims(img2_array, 0)

    # pass images through model to get output image
    output = model.predict([img1_tensor, img2_tensor])
    output_image = Image.fromarray((output[0] * 255).astype(np.uint8))

    # save output image to disk
    output_image.save("output_image.jpg")

    # return output image
    return send_file("output_image.jpg", mimetype="image/jpeg")


if __name__ == "__main__":
    app.run(port=8080, debug=True)

model.save("model/model_20k.h5")