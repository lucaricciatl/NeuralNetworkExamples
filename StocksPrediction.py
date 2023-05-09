import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Input, SimpleRNN, LSTM, GRU, GlobalMaxPool1D, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam

from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np

data = pd.read_csv("dataset/NVDA.csv")
df = pd.DataFrame(data)

print(df)

headers = list(df.columns.values)

print(headers)


#builld the dataset 
series = df.get("Open").values.reshape(-1,1)
#plt.plot(series)
sclaler = StandardScaler()
sclaler.fit(series[:len(series)//2])
series = sclaler.transform(series).flatten()

#build the dataset
T=10
X = []
Y = []
for t in range(len(series)-T):
  X.append(series[t:t+T])
  Y.append(series[t+T])
X = np.asarray(X).reshape(-1, T , 1)
Y = np.asarray(Y)
n = len(X)
print(X.shape)
print(Y.shape)
print(n)


#build the model
i = Input(shape=(T,1))
l = LSTM(50)(i)
x = Dense(50)(l)
x = Dense(50)(x)
x = Dense(1)(x)
model = Model(i,x)
model.compile(loss="mse", optimizer=Adam(lr=0.3))

#train the model
r = model.fit(X[:-n//2], Y[:-n//2], validation_data=(X[-n//2:], Y[-n//2:]), epochs=100)

#plot the loss
#figure = plt.figure(1)
#plt.plot(r.history["loss"], label="loss")
#plt.plot(r.history["val_loss"], label="val_loss")
#plt.legend()


#one-step forecast
figure = plt.figure(3)
outputs = model.predict(X)
print(outputs.shape)
predictions = outputs[:,0]
plt.plot(Y, label="targets")
plt.plot(predictions, label="predictions")

#plot the prediction
validation_target = Y[-n//2:]
validation_prediction = []
i = -n//2
while len(validation_prediction) < len(validation_target)-3:
    p = model.predict(X[i+3].reshape(1,-1,1))[0,0]
    i += 1
    validation_prediction.append(p)

plt.figure(2)
plt.plot(validation_target, label="forecast target")
plt.plot(validation_prediction, label="forecast prediction")
plt.legend()