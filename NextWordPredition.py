import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Input, SimpleRNN, LSTM, GRU, GlobalMaxPool1D, Conv1D, MaxPooling1D, Flatten, Embedding, Bidirectional, Dropout,BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import platform
import sys
import sklearn as sk
import scipy as sp

def print_info():
    print(f"Python Platform: {platform.platform()}")
    print(f"Tensor Flow Version: {tf.__version__}")
    print(f"Keras Version: {tf.keras.__version__}")
    print()
    print(f"Python {sys.version}")
    print(f"Pandas {pd.__version__}")
    print(f"Scikit-Learn {sk.__version__}")
    print(f"SciPy {sp.__version__}")
    gpu = len(tf.config.list_physical_devices('GPU')) > 0
    print("GPU is", "available" if gpu else "NOT AVAILABLE")

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.show()

def generate_model(num_words, max_sequence_len):
    model = Sequential()
    model.add(Embedding(num_words, 200, input_length=max_sequence_len-1))
    model.add(Dense(100))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(100, return_sequences=True)))
    model.add(Bidirectional(LSTM(50)))
    model.add(Dense(num_words, activation='softmax'))
    adam = Adam(lr=0.015)
    model.compile(loss='categorical_crossentropy',
              optimizer=adam, metrics=['accuracy'])
    return model 


def fit_and_save():
    with tf.device('/device:GPU:0'):
        history = model.fit(xs, ys, epochs=50, verbose=1, batch_size=32, validation_split=0.1)
        model.save('./models/nwp.h5')
    return history


def evaluate(tokenizer, seed_text, next_words):
    max_sequence_len = max([len(x) for x in input_sequences])
    for i in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences(
            [token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            output_word = word
            seed_text += " " + output_word
            break
    print(seed_text)

print_info()

# load the csv file
data = pd.read_csv("dataset/medium_data.csv")
data = pd.DataFrame(data)
print(data)

data.head()

# build the dataset
print(data.shape)
data['title'] = data['title'].apply(lambda x: x.replace(u'\xa0', u' '))
data['title'] = data['title'].apply(lambda x: x.replace('\u200a', ' '))

tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(data['title'])

num_words = len(tokenizer.word_index) + 1

print("Total number of words: ", num_words)
#print("Word_index: ", tokenizer.word_index)

input_sequences = []
for line in data['title']:
    token_list = tokenizer.texts_to_sequences([line])[0]
    #print(token_list)

    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# print(input_sequences)
print("Total input sequences: ", len(input_sequences))

# pad sequences
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(
    input_sequences, maxlen=max_sequence_len, padding='pre'))
input_sequences[1]

# create features and label
xs, labels = input_sequences[:, :-1], input_sequences[:, -1]
ys = tf.keras.utils.to_categorical(labels, num_classes=num_words)

model = generate_model(num_words, max_sequence_len)

print(model.summary())

history = fit_and_save()

plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')


print(model)

