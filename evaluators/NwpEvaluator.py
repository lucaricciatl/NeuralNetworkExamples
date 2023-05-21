import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Input, SimpleRNN, LSTM, GRU, GlobalMaxPool1D, Conv1D, MaxPooling1D, Flatten, Embedding, Bidirectional
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd


def evaluate_next(tokenizer, seed_text, next_words,input_sequences):
    max_sequence_len = max([len(x) for x in input_sequences])
    for i in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences(
            [token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = np.argmax(model.predict(token_list), axis=-1)
        print(predicted)
        output_word = ""
        #get the element index of mapping 
        items = tokenizer.word_index.items()
        for word, index in items:
            if index == predicted:
                output_word = word
        break
    print(output_word)


# load the csv file
dir_path = os.path.dirname(os.path.realpath(__file__))
data = pd.read_csv(dir_path+"/../dataset/medium_data.csv")
data = pd.DataFrame(data)
print(data)

data.head()
data['title'] = data['title'].apply(lambda x: x.replace(u'\xa0', u' '))
data['title']=data['title'].apply(lambda x: x.replace('\u200a', ' '))
model = tf.keras.models.load_model(dir_path+'/../models/nwp.h5')
model.load_weights(dir_path+'/../models/nwp.h5')

seed_text = "artificial"
next_words = 7

# For those words which are not found in word_index
tokenizer = Tokenizer(20000)
tokenizer.fit_on_texts(data['title'])
input_sequences = []


print("Word_index: ", tokenizer.word_index)

for line in data['title']:
    token_list = tokenizer.texts_to_sequences([line])[0]
    # print(token_list)

    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

total_words = len(tokenizer.word_index) + 1
max_sequence_len = max([len(x) for x in input_sequences])

evaluate_next(tokenizer,seed_text,next_words,input_sequences)