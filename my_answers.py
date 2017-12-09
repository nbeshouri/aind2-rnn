import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras
import string


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    X = []
    y = []
    for i in range(len(series) - window_size):
        X.append(series[i:window_size + i])
        y.append(series[window_size + i])
    
    # Convert to numpy arrays.
    X = np.asarray(X)
    y = np.asarray(y)
    y.shape = (len(y), 1)

    return X, y


# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(units=5, input_shape=(window_size, 1), unroll=True))
    model.add(Dense(units=1, activation=None))
    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = {'!', ',', '.', ':', ';', '?'}
    to_remove = set(text) - set(string.ascii_lowercase) - punctuation
    for char in to_remove:
        text = text.replace(char, ' ')
    return text


### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    X = []
    y = []
    for i in range(0, len(text) - window_size, step_size):
        X.append(text[i:window_size + i])
        y.append(text[window_size + i])
    
    return X, y


# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(units=200, input_shape=(window_size, num_chars), unroll=True))
    model.add(Dense(units=num_chars, activation='softmax'))
    return model
