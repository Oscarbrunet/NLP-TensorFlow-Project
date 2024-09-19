import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from Model import CustomModel
import matplotlib.pyplot as plt

# Charger les données prétraitées
train_padded = np.load('train_padded.npy')
train_labels = np.load('train_labels.npy')
val_padded = np.load('val_padded.npy')
val_labels = np.load('val_labels.npy')

# Charger max_length et num_unique_words depuis le fichier texte
with open('model_params.txt', 'r') as f:
    params = f.readlines()
    max_length = int(params[0].split(': ')[1])
    num_unique_words = int(params[1].split(': ')[1])

# Initialiser le modèle
model = CustomModel(num_unique_words=num_unique_words, max_length=max_length)
model.build((None, max_length))  # Construire le modèle pour l'entraînement

# Définir la fonction de perte, l'optimiseur et les métriques
loss = tf.keras.losses.SparseCategoricalCrossentropy()
optim = Adam(learning_rate=0.001)
metrics = ["accuracy"]

# Compiler le modèle
model.compile(optimizer=optim, loss=loss, metrics=metrics)

# Entraîner le modèle
history = model.fit(train_padded, train_labels, epochs=5, validation_data=(val_padded, val_labels), verbose=2)

# Tracer la perte (loss)
plt.figure(figsize=(12, 5))

# Perte d'entraînement
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Loss (train)')
plt.plot(history.history['val_loss'], label='Loss (val)')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Précision (accuracy)
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Accuracy (train)')
plt.plot(history.history['val_accuracy'], label='Accuracy (val)')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Sauvegarder les graphiques
plt.savefig('training_curves.png')

# Afficher les graphiques
plt.show()

# Sauvegarder le modèle
model.save('trained_model.keras')
