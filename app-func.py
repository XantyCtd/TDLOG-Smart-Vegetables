from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)

model = tf.keras.models.load_model("model.hdf5")

img_height = 180
img_width = 180


@app.route("/")
def index():
    return render_template("upload.html")


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["image"]
    if file:
        # save the image to disk
        file.save("image.jpg")

        # read the image and treat it using OpenCV
        img = tf.keras.utils.load_img("image.jpg", target_size=(img_height, img_width))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        # save the treated image to disk
        class_names = [
            "Haricot",
            "Melon amer",
            "Calebasse",
            "Brinjal",
            "Brocoli",
            "Chou",
            "Poivron",
            "Carotte",
            "Chou-fleur",
            "Concombre",
            "Papaye",
            "Pomme de terre",
            "Citrouille",
            "Radis blanc",
            "Tomate",
        ]

        os.remove("image.jpg")

        return "This image most likely belongs to {} with a {:.2f} percent confidence.".format(
            class_names[np.argmax(score)], 100 * np.max(score)
        )


if __name__ == "__main__":
    app.run()
