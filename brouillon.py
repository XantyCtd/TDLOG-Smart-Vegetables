from flask import Flask
from flask import render_template
import os

SmartVegetables = Flask(__name__)
IMG_FOLDER = os.path.join("static", "IMG")
SmartVegetables.config["UPLOAD_FOLDER"] = IMG_FOLDER


@SmartVegetables.route("/")
def texte():
    return render_template("texte.html", message="Bienvenue sur Smart Vegetables")


def Display_IMG():
    Flask_Logo = os.path.join(SmartVegetables.config["UPLOAD_FOLDER"], "flask-logo.png")
    return render_template("index.html", user_image=Flask_Logo)


@SmartVegetables.route("/next")
def suite():
    return render_template(
        "page_suivante.html", message2="Choisisser une image à télécharger"
    )


if __name__ == "__main__":  # calling  main
    SmartVegetables.debug = (
        True  # setting the debugging option for the application instance
    )
    SmartVegetables.run()
