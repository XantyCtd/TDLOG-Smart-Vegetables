from flask import url_for
from flask import Flask, request, render_template
import tensorflow as tf
import os
from flask_login import (
    login_user,
    LoginManager,
    login_required,
    current_user,
    logout_user,
)
from werkzeug.utils import redirect
import numpy as np
from flask_bcrypt import Bcrypt
import database
from config import class_names

"""Initialisation de l'application"""

app = Flask(__name__)
app.debug = True
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
app.config["SECRET_KEY"] = "you-will-never-guess"
bcrypt = Bcrypt(app)
login_manager = LoginManager()
bcrypt = Bcrypt(app)
login_manager = LoginManager()
login_manager.init_app(app)
IMG_FOLDER = os.path.join("templates", "IMG")
app.config["UPLOAD_FOLDER"] = IMG_FOLDER
model = tf.keras.models.load_model("data/2_prediction/model.hdf5")
img_height = 180
img_width = 180

# Définitions des pages de l'application
@app.route("/")
def accueil():
    """création de la page d'accueil de l'application"""
    return render_template("accueil.html", message="Bienvenue sur Smart Vegetables")


@app.route("/About")
def About():
    """création de la page à propos de l'application"""
    return render_template("About.html")


@login_manager.user_loader
def load_user(user_id):
    return database.User.get(user_id)


@login_manager.unauthorized_handler
# si l'utilisateur tente d'accéder à une page alors qu'il n'est pas connecté


def unauthorized():
    return "You must be logged in to access this page."


@app.route("/signup", methods=["GET", "POST"])
def signup():
    """Définition de la page inscription, qui permet à l'utilisateur de s'inscrire sur le site"""
    if request.method == "POST":
        # Create a new user with the provided email and password
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]
        hashed_password = bcrypt.generate_password_hash(password).decode("utf-8")
        database.User.create(
            username=username,
            email=email,
            password=hashed_password,
            first_name=request.form["first_name"],
            last_name=request.form["last_name"],
        )
        return redirect(url_for("SignIn"))
    return render_template("register.html")


@app.route("/SignIn", methods=["GET", "POST"])
def SignIn():
    """Définition de la page connexion, qui permet à l'utilisateur de se connecter"""
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        try:
            user = database.User.get(database.User.email == email)
            # check if the password is correct
            if bcrypt.check_password_hash(user.password, password):
                login_user(user)
                return redirect(url_for("profile"))
            else:
                return "Incorrect password"
        except database.User.DoesNotExist:
            return "Incorrect email"
    return render_template("login2.html")


@app.route("/logout")
def logout():
    """Définition de la page déconnexion,
    qui permet à l'utilisateur de se déconnecter"""
    logout_user()
    return redirect(url_for("SignIn"))


@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    """Définition de la page profil qui donne accès au profil
    une fois que l'utilisateur est connecté"""
    return render_template("profile.html")


@app.route("/suite", methods=["GET", "POST"])
@login_required
# def suite():
def index1():
    """Définition de la page image à télécharger, on peut y accéder
    une fois que l'utilisateur est connecté"""
    return render_template(
        "page_suivante.html", message2="Choisissez une image à télécharger"
    )


@app.route("/upload", methods=["POST"])
@login_required
def upload():
    """Définition de la page image à télécharger, on peut y accéder une fois que l'utilisateur est connecté"""
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

        os.remove("image.jpg")
        return render_template(
            "page_suivante2.html",
            message3="Le légume associé à l'image est {} avec une confiance de {:.2f}%".format(
                class_names[np.argmax(score)], 100 * np.max(score)
            ),
        )


def Display_IMG():
    Flask_Logo = os.path.join(app.config["UPLOAD_FOLDER"], "static/images/logo.png")
    return render_template("page_suivante2.html", user_image=Flask_Logo)


if __name__ == "__main__":
    # lancement de l'application
    app.run()
