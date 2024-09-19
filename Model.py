import tensorflow as tf
from tensorflow.keras import layers

class CustomModel(tf.keras.Model):
    def __init__(self, num_unique_words, max_length):
        super(CustomModel, self).__init__()
        self.embedding = layers.Embedding(input_dim=num_unique_words, output_dim=32)
        self.lstm = layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2) 
        self.dropout = layers.Dropout(0.5)  
        self.dense = layers.Dense(3, activation='softmax')

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.lstm(x)
        x = self.dropout(x)  
        x = self.dense(x)
        return x

    def build(self, input_shape):
        self.embedding.build(input_shape)
        self.lstm.build(self.embedding.compute_output_shape(input_shape))
        self.dropout.build(self.lstm.compute_output_shape(self.embedding.compute_output_shape(input_shape)))
        self.dense.build(self.lstm.compute_output_shape(self.embedding.compute_output_shape(input_shape)))
        self.built = True
