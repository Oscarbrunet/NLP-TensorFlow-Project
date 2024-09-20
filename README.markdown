## Table of Contents
1. [Description]
2. [Installation]
3. [Usage]

## Description

_In order to reach the objective, I downloaded a database from Kaggle. You can see the data thanks to `Visualization_Data.py`._

After I implemented a script to process the data `Data_processing.py`. More precisely this script:

- Loads `train.csv` into a pandas DataFrame
- Defines functions to:
    - Remove URLs from text.
    - Remove punctuation while keeping asterisks.
    - Remove common English stopwords from the text.
- Defines a function to count word occurrences in text.
- Creates a global counter to aggregate word counts across all text data.
- Converts sentiment labels (negative, neutral, positive) to numerical values (0, 1, 2).
- Splits the DataFrame into training and validation sets (80% training, 20% validation).
- Uses TensorFlow’s Tokenizer to convert text to sequences of integers.
- Pads the sequences to ensure uniform input size for the model.
- Saves the padded sequences and labels to .npy files for use in training.
- Saves model parameters (e.g., max_length and num_unique_words) to a text file.

Then I implemented a `Model.py` to define the model of the neuronal network. It includes:

The `CustomModel` class defines a Keras model for text classification.

1. **Embedding Layer**: Transforms integer sequences into dense vectors.
2. **LSTM Layer**: Processes sequences with 64 units and applies dropout and recurrent dropout for regularization.
3. **Dropout Layer**: Adds a 50% dropout to the LSTM’s output to prevent overfitting.
4. **Dense Layer**: Outputs class probabilities with a softmax activation function.

The `call` method performs the forward pass, applying the layers sequentially. The `build` method sets up layer shapes and dependencies. This model is suitable for tasks like sentiment analysis or text categorization.

The model_summary.py enables use to see the number of parameters of the model. 

Finally, the train_model.py enables to run the model and creates charts to analyse the result. 


## Installation

clone the project 
create an virtual environment 
install the required packages (pip install -r requirements.txt
)
Download NLTK Data (import nltk
nltk.download('stopwords') )

## Usage

Run 
1) Vizualization_Data.py
2) Data_processing.py
3) Model.py
4) model.summary.py
5) train_model.py
