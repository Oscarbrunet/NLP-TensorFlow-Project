import tensorflow as tf
import numpy as np

# Charger les données prétraitées
train_padded = np.load('train_padded.npy')
train_labels = np.load('train_labels.npy')

# Charger max_length et num_unique_words depuis le fichier texte
with open('model_params.txt', 'r') as f:
    params = f.readlines()
    max_length = int(params[0].split(': ')[1])
    num_unique_words = int(params[1].split(': ')[1])
    
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow import keras

# Créer un modèle séquentiel
model = keras.models.Sequential()

# Ajouter une couche d'embedding
model.add(layers.Embedding(input_dim=num_unique_words, output_dim=32, input_length=max_length))

# Ajouter une couche LSTM avec dropout pour éviter l'overfitting
model.add(layers.LSTM(64, dropout=0.1))

# Ajouter une couche Dense avec activation softmax pour la classification multi-classes
model.add(layers.Dense(3, activation="softmax"))  # 3 classes de sortie (0, 1, 2)

# Résumer la structure du modèle
model.summary()

# Compiler le modèle avec une fonction de perte multi-classes
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
optim = keras.optimizers.Adam(learning_rate=0.001)
metrics = ["accuracy"]

# Compilation du modèle
model.compile(loss=loss, optimizer=optim, metrics=metrics)

# Entraîner le modèle sur les données padded
model.fit(train_padded, train_labels, epochs=20, validation_data=(val_padded, val_labels), verbose=2)

# Faire des prédictions
predictions = model.predict(train_padded)

# Convertir les prédictions en classes (0, 1 ou 2) en prenant l'indice de la classe avec la probabilité la plus élevée
predictions = [tf.argmax(p).numpy() for p in predictions]
