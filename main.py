import tensorflow as tf
import pandas as pd

# Ajuster le nombre de colonnes affichées
pd.set_option('display.max_columns', None)

# Ajuster le nombre de lignes affichées
pd.set_option('display.max_rows', None)

# Ajuster la largeur des colonnes affichées
pd.set_option('display.max_colwidth', None)

df = pd.read_csv("train.csv", encoding="ISO-8859-1")

print("shape =", df.shape)
print(df.head())