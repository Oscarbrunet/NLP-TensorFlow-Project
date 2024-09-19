import tensorflow as tf
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from collections import Counter
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Téléchargement des stopwords si ce n'est pas encore fait
nltk.download('stopwords')

# Définition des stopwords
stop = set(stopwords.words("english"))

df = pd.read_csv("train.csv", encoding="ISO-8859-1")

# Supprimer les colonnes 1, 4, 5, 6, 7, 8, 9
df = df.drop(df.columns[[1, 4, 5, 6, 7, 8, 9]], axis=1)

def remove_URL(text):
    if isinstance(text, str):
        url = re.compile(r"https?://\S+|www\.\S+")
        return url.sub(r"", text)
    else:
        return text

def remove_punct_keep_asterisk(text):
    if isinstance(text, str):
        translator = str.maketrans("", "", string.punctuation.replace("*", ""))
        return text.translate(translator)
    else:
        return text

def remove_stopwords(selected_text):
    if isinstance(selected_text, str):
        filtered_words = [word.lower() for word in selected_text.split() if word.lower() not in stop]
        return " ".join(filtered_words)
    else:
        return selected_text

df["selected_text"] = df.selected_text.map(remove_URL) 
df["selected_text"] = df.selected_text.map(remove_punct_keep_asterisk)
df["selected_text"] = df.selected_text.map(remove_stopwords)

def counter_word(text):
    if isinstance(text, str):
        return Counter(word.lower() for word in text.split())
    else:
        return Counter()

# Création d'un compteur global
global_counter = Counter()

for text in df["selected_text"]:
    global_counter += counter_word(text)

#print(global_counter)
#print(global_counter.most_common(10))


#itère sur la colonne sentiment de df, si la valeur est negative, on la remplace par 0, si c'est neutral on la remplace par 1 et si c'est positive on la remplace par 2
df["sentiment"] = df["sentiment"].map({"negative": 0, "neutral": 1, "positive": 2})

train_size = int(df.shape[0] * 0.8)

train_df = df[:train_size]
val_df = df[train_size:]

train_sentences = train_df.selected_text.to_numpy()
train_sentences = [str(sentence) for sentence in train_sentences]
train_labels = train_df.sentiment.to_numpy()

val_sentences = val_df.selected_text.to_numpy()
val_sentences = [str(sentence) for sentence in val_sentences]
val_labels = val_df.sentiment.to_numpy()


num_unique_words = len(global_counter)
# Ensure all elements in train_sentences are strings

# Now fit the tokenizer
tokenizer = Tokenizer(num_words=num_unique_words)
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index
train_sequences = tokenizer.texts_to_sequences(train_sentences)
val_sequences = tokenizer.texts_to_sequences(val_sentences)

max_length = 20

train_padded = pad_sequences(train_sequences, maxlen=max_length, padding="post", truncating="post")
val_padded = pad_sequences(val_sequences, maxlen=max_length, padding="post", truncating="post")
np.save('train_padded.npy', train_padded)
np.save('train_labels.npy', train_labels)
np.save('val_padded.npy', val_padded)
np.save('val_labels.npy', val_labels)


with open('model_params.txt', 'w') as f:
    f.write(f'max_length: {max_length}\n')
    f.write(f'num_unique_words: {num_unique_words}\n')
