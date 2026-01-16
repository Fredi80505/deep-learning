📊Interface streamlit pour des prédictions interactives







🎯Objectif: Cette interface a été développée, avec pour but d'effectuer des prédictions interactives, visualiser les résultats de classification, effectuer la demonstration et l'évaluation du modèle en local.



📁 Organisation des dossiers et fichiers du projet:





Projet BARGO Alfred/

├─ TP 2/

│  ├─ model2/

│  │  ├─ mlp\_model.joblib

│  ├─ TP 2 Diagnostique\_cancer.ipynb

├─ TP 3/

│  ├─ model3/

│  │  ├─ mnist\_model.joblib

│  │  ├─ model\_metadata.pkl

│  ├─ app\_stream3.py

│  ├─ TP 3 Classification\_images manuscrites.ipynb

│  ├─ requirements.txt

├─ TP 4/

│  ├─ model4/

│  │  ├─ cifar\_model.joblib

│  │  ├─ model\_cifar\_metadata.pkl

│  ├─ app\_stream4.py

│  ├─ TP 4 Classification\_image en couleur.ipynb

│  ├─ requirements.txt

├─ README.txt





⚙️Prérequis:

S'assurer d'avoir déjà installer python>=3.8 
S'assurer d'avoir un navigateur web (Chrome, Microsoft Edge, Firefox ou autres)





📦Installation des dépendances



Pour chaque TP: 

&nbsp;  ouvrir le terminal Anaconda Prompt; 

&nbsp;  activer l'environnement contenant Tensorflow/keras;

&nbsp;  aller dans le dossier correspondant et taper la commande suivante: *pip install -r requirements.txt*







▶️Lancement de l’interface Streamlit



Toujours dans le terminal Anaconda Prompt:

&nbsp;  se placer dans le dossier du TP correspondant; 

&nbsp;  activer l'environnement contenant Tensorflow/keras;


&nbsp;  puis exécuter la commande suivante: streamlit run app\_stream3.py ou streamlit run app\_stream4.py selon que ce soit le TP 3 ou le TP 4 qui est le dossier courant.


Par la suite, une page web s'ouvrira automatiquement dans le navigateur





🧠Utilisation de l'interface



Dans l'interface qui s'affiche, il est possible de charger une image à partir du stockage interne et ensuite cliquer sur le bouton de prédiction pour enfin voir les résultat de prédiction du modèle.


*Auteur*: **BARGO Alfred**
