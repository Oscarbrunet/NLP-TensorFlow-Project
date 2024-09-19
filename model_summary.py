import numpy as np
from Model import CustomModel  # Assure-toi que le fichier Model.py est dans le même répertoire
import tensorflow as tf

# Charger les données prétraitées
train_padded = np.load('train_padded.npy')
train_labels = np.load('train_labels.npy')

# Charger max_length et num_unique_words depuis le fichier texte
with open('model_params.txt', 'r') as f:
    params = f.readlines()
    max_length = int(params[0].split(': ')[1])
    num_unique_words = int(params[1].split(': ')[1])

# Initialiser le modèle
model = CustomModel(num_unique_words=num_unique_words, max_length=max_length)
model.build((None, max_length))  # Construire le modèle pour afficher le résumé

# Afficher le résumé du modèle
model.summary()
