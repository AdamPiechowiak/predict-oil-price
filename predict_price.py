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
from apscheduler.schedulers.background import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
import datetime
 

# tworzenie sekwencji danych wejściowych i wyjściowych
def create_sequences(data, seq_length):
    
    X = []
    y = []
    for i in range(len(data)-seq_length):
        X.append(data[i:i+seq_length, :])
        y.append(data[i+seq_length, :]) # wyjście - pierwsza kolumna danych
    return np.array(X), np.array(y)

def predict(df,predict_n_days):
    
    data = df.to_numpy()
    
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

    seq_length = 100 # długość sekwencji

    X_train, y_train = create_sequences(train_data_norm, seq_length)
    X_val, y_val = create_sequences(scaler.transform(val_data), seq_length)
    X_test, y_test = create_sequences(scaler.transform(test_data), seq_length)

    input_tensor = output_tensor = Input(X_train.shape[1:])
    output_tensor = TimeDistributed(Dense(32, activation = 'selu'))(output_tensor)
    output_tensor = TimeDistributed(Dense(32, activation = 'selu'))(output_tensor)
    output_tensor = TimeDistributed(Dropout(0.1))(output_tensor)
    output_tensor = LSTM(1, activation = 'selu', return_sequences = False)(output_tensor)

    model = Model(inputs = input_tensor, outputs = output_tensor)
    model.compile(optimizer='RMSProp', loss='MeanSquaredError')
    
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

        y_pred = model_list[i].predict(X_test)
            
        diff_pred = np.diff(y_pred.squeeze())
        diff_true = np.diff(y_test[:,i])
        diff_pred = np.sign(diff_pred)
        diff_true = np.sign(diff_true)
            
        acc_list.append(accuracy_score(diff_true, diff_pred))

    print(acc_list)
    
    #przewidywanie cen 
    predict_data = np.array(data[:])

    for i in range(predict_n_days):
        to_predict = scaler.transform(predict_data[-100:])
        to_predict = to_predict.reshape((1,)+to_predict.shape)
        
        predict_list = []
        for j in range(len(model_list)):
            predict_list.append(model_list[i].predict(to_predict))
        
        p=np.array(predict_list)

        p = p.reshape((1,len(model_list)))

        p2 = scaler.inverse_transform(p)
        predict_data = np.append(predict_data,p2,axis=0)
        
    date = np.array(df.index)
    for i in range(predict_n_days):
        date = np.append(date, date[len(date)-5]+np.timedelta64(7,'D'))

    pd_predict_data = pd.DataFrame(predict_data, columns = ['Open','High','Low','Close','Adj Close','Volume'])
    pd_predict_data.index = date
    return pd_predict_data


def job(): 
    
    
    if(yf.Ticker("CL=F").history(period='1d').empty):
        
        
        oil = pd.read_csv("oil_price.csv",parse_dates=True,index_col=0)
        dates = oil.index.tolist()
        
        i=0
        
        while(dates[-1]-np.datetime64('today')<np.timedelta64(7,'D')):
            d = dates[-5]+np.timedelta64(7,'D')
            dates = np.append(dates, d)
            i+=1
        
        predict_data = predict(oil,i)
        
        predict_data.to_csv("oil_price.csv")
        
        
    else:
        
        # Pobierz historyczne ceny ropy naftowej WTI Crude Oil
        oil = yf.download("CL=F")
        print(oil)
        
        predict_data = predict(oil,5)
        
        predict_data.to_csv("oil_price.csv")
        
        

sched = BlockingScheduler()

sched.add_job(job, CronTrigger( hour=10))

sched.start()


 









    