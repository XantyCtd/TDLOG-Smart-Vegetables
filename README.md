# TDLOG-Smart-Vegetables
Projet TDLOG 

Abigaïl Lepers,
Noémie Dufourd,
Stéfanie Moara Bastos Sousa Bomfim Fraga et
Alexandre Chotard

# Description
Application de reconnaissance de légumes par machine learning

Entrée de l'application : image à uploader (présentation sous la forme de plateforme web)

L'application retourne le nom du légume

# Lancement de l'application
Pour lancer l'application, il faut se rendre dans le fichier app.py et lancer le programme, on aura ainsi le lien envoyant vers le navigateur qui apparaitra. 

Ensuite, un nouvel utilisateur doit s'inscrire pour avoir accès aux différents points de l'application. Une fois cela fait, il pourra scanner un légume de son choix et recevoir une information lui indiquant de quel légume il s'agit.

# Outils utilisés

Machine learning : Tensorflow

Interface web : Flask

# Description des fichiers
Il existe dans cette application différents répertoires : 

-Le repertoire static contenant un fichier css avec le style de l'application et un fichier images avec les différentes images utilisées pour créer l'application.

-Le repertoire templates contenant tout les fichiers html des différentes pages de l'application.

-Le répertoire data contenant les images et les données collectées au fur et à mesure de l'entrainement du modèle

-Le répertoire src contenant les fichiers python à lancer pour préparer, entraîner et évaluer le modèle de deep larning

-Un fichier app.py responsable de la définition des pages de l'application et du lancement de cette dernière.

-Un fichier config.py qui contient les noms des différents légumes reconnaissables par l'application

-Des fichier database.py et user.db relatif à la base de données

