import yfinance as yf
import datetime
import pandas as pd
import numpy as np
from keras.layers import GRU, Input, LSTM, Dropout
from keras.models import Model
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from keras.layers import TimeDistributed, Dense
from keras.utils.vis_utils import plot_model

#last_year = str(datetime.datetime.now().year-1) + "-" + str(datetime.datetime.now().month) + "-" + str(datetime.datetime.now().day)
#today = str(datetime.datetime.now().year) + "-" + str(datetime.datetime.now().month) + "-" + str(datetime.datetime.now().day)
#print(today)


# Pobierz historyczne ceny ropy naftowej WTI Crude Oil
oil = yf.download("CL=F")#), start=last_year, end=today)
data = oil.to_numpy()


# podział danych na zbiór treningowy, walidacyjny i testowy
train_size = int(len(data) * 0.6)
val_size = int(len(data) * 0.2)
test_size = len(data) - train_size - val_size

train_data = data[:train_size, :]
val_data = data[train_size:train_size+val_size, :]
test_data = data[train_size+val_size:, :]


# normalizacja danych treningowych
scaler = StandardScaler()
train_data_norm = scaler.fit_transform(train_data)

# tworzenie sekwencji danych wejściowych i wyjściowych
def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data)-seq_length):
        X.append(data[i:i+seq_length, :])
        y.append(data[i+seq_length, 0]) # wyjście - pierwsza kolumna danych
    return np.array(X), np.array(y)

seq_length = 100 # długość sekwencji

X_train, y_train = create_sequences(train_data_norm, seq_length)
X_val, y_val = create_sequences(scaler.transform(val_data), seq_length)
X_test, y_test = create_sequences(scaler.transform(test_data), seq_length)


print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)
print(X_test.shape)
print(y_test.shape)

input_tensor = output_tensor = Input(X_train.shape[1:])
output_tensor = TimeDistributed(Dense(32, activation = 'selu'))(output_tensor)
output_tensor = TimeDistributed(Dense(32, activation = 'selu'))(output_tensor)
output_tensor = TimeDistributed(Dropout(0.1))(output_tensor)
output_tensor = LSTM(1, activation = 'selu', return_sequences = False)(output_tensor)

model = Model(inputs = input_tensor, outputs = output_tensor)
model.compile(optimizer='RMSProp', loss='MeanSquaredError')

plot_model(model)
print(model.summary())
model.fit(x = X_train, y = y_train, batch_size=32, epochs=20, validation_data=(X_val, y_val))


y_pred = model.predict(X_test)

x = np.arange(y_pred.shape[0])
plt.figure(1)
plt.subplot(211)
plt.plot(x, y_pred, label = 'Predykcje')
plt.plot(x, y_test, label = 'Wartości prawdziwe')
plt.legend()
diff = y_pred.squeeze()-y_test
plt.subplot(212)
plt.plot(x, diff, label = 'różnice')
plt.legend()

diff_pred = np.diff(y_pred.squeeze())
diff_true = np.diff(y_test)
diff_pred = np.sign(diff_pred)
diff_true = np.sign(diff_true)
print(confusion_matrix(diff_true, diff_pred))
print(accuracy_score(diff_true, diff_pred))

to_predict = scaler.transform(data[-100:])
to_predict = to_predict.reshape((1,)+to_predict.shape)
predict = model.predict(to_predict)