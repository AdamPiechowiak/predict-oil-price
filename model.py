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
import copy

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
        y.append(data[i+seq_length, :]) # wyjście - pierwsza kolumna danych
    return np.array(X), np.array(y)

seq_length = 100 # długość sekwencji

X_train, y_train = create_sequences(train_data_norm, seq_length)
X_val, y_val = create_sequences(scaler.transform(val_data), seq_length)
X_test, y_test = create_sequences(scaler.transform(test_data), seq_length)


# print(X_train.shape)
# print(y_train.shape)
# print(X_val.shape)
# print(y_val.shape)
# print(X_test.shape)
# print(y_test.shape)


input_tensor = output_tensor = Input(X_train.shape[1:])
output_tensor = TimeDistributed(Dense(32, activation = 'selu'))(output_tensor)
output_tensor = TimeDistributed(Dense(32, activation = 'selu'))(output_tensor)
output_tensor = TimeDistributed(Dropout(0.1))(output_tensor)
output_tensor = LSTM(1, activation = 'selu', return_sequences = False)(output_tensor)

model = Model(inputs = input_tensor, outputs = output_tensor)
model.compile(optimizer='RMSProp', loss='MeanSquaredError')

plot_model(model)
# print(model.summary())

# print(y_train[:,0].shape)

open_model = copy.deepcopy(model)
high_model =  copy.deepcopy(model)
low_model = copy.deepcopy(model)
close_model = copy.deepcopy(model)
adj_model = copy.deepcopy(model)
volume_model = copy.deepcopy(model)


acc_list = []
model_list = [open_model,high_model,low_model,close_model,adj_model,volume_model]

#trenowanie modeli i obliczenie dokladnosci
for i in range(len(model_list)):
    model_list[i].fit(x = X_train, y = y_train[:,i], batch_size=32, epochs=20, validation_data=(X_val, y_val[:,i]))

# open_model.fit(x = X_train, y = y_train[:,0], batch_size=32, epochs=20, validation_data=(X_val, y_val[:,0]))
# high_model.fit(x = X_train, y = y_train[:,1], batch_size=32, epochs=20, validation_data=(X_val, y_val[:,1]))
# low_model.fit(x = X_train, y = y_train[:,2], batch_size=32, epochs=20, validation_data=(X_val, y_val[:,2]))
# close_model.fit(x = X_train, y = y_train[:,3], batch_size=32, epochs=20, validation_data=(X_val, y_val[:,3]))
# adj_model.fit(x = X_train, y = y_train[:,4], batch_size=32, epochs=20, validation_data=(X_val, y_val[:,4]))
# volume_model.fit(x = X_train, y = y_train[:,5], batch_size=32, epochs=20, validation_data=(X_val, y_val[:,5]))

    y_pred = model_list[i].predict(X_test)
        
    diff_pred = np.diff(y_pred.squeeze())
    diff_true = np.diff(y_test[:,i])
    diff_pred = np.sign(diff_pred)
    diff_true = np.sign(diff_true)
    #print(confusion_matrix(diff_true, diff_pred))
    #print(accuracy_score(diff_true, diff_pred))
        
    acc_list.append(accuracy_score(diff_true, diff_pred))

print(acc_list)


#przewidywanie cen dla następnego tygodnia
predict_data = np.array(data[:])

for i in range(5):
    to_predict = scaler.transform(predict_data[-100:])
    to_predict = to_predict.reshape((1,)+to_predict.shape)
    predict_o = open_model.predict(to_predict)
    predict_h = high_model.predict(to_predict)
    predict_l = low_model.predict(to_predict)
    predict_c = close_model.predict(to_predict)
    predict_a = adj_model.predict(to_predict)
    predict_v = volume_model.predict(to_predict)
    
    p=np.array([predict_o,predict_h,predict_l,predict_c,predict_a,predict_v])

    p = p.reshape((1,6))

    p2 = scaler.inverse_transform(p)
    predict_data = np.append(predict_data,p2,axis=0)
    


for i in range(len(model_list)):
    x = range(len(predict_data[-30:,i]))
    plt.figure(i)
    plt.plot(x, predict_data[-30:,i], label = 'Predykcja '+str(i))
    plt.legend()
'''
y_pred = volume_model.predict(X_test)


#y_pred = scaler.inverse_transform(y_pred)
#y_test = scaler.inverse_transform(y_test)

x = np.arange(y_pred.shape[0])
plt.figure(1)
plt.subplot(211)
plt.plot(x, y_pred, label = 'Predykcje')
plt.plot(x, y_test[:,5], label = 'Wartości prawdziwe')
plt.legend()
diff = y_pred.squeeze()-y_test[:,5]
plt.subplot(212)
plt.plot(x, diff, label = 'różnice')
plt.legend()

diff_pred = np.diff(y_pred.squeeze())
diff_true = np.diff(y_test[:,5])
diff_pred = np.sign(diff_pred)
diff_true = np.sign(diff_true)
print(confusion_matrix(diff_true, diff_pred))
print(accuracy_score(diff_true, diff_pred))'''