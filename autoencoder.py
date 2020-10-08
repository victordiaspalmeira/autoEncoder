import numpy as np
import tensorflow as tf
import pandas as pd
pd.options.mode.chained_assignment = None
import seaborn as sns
import os
from matplotlib.pylab import rcParams
import matplotlib.pyplot as plt
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dropout, RepeatVector, TimeDistributed, Dense
import pickle as p

rcParams['figure.figsize'] = 14, 8
sns.set(style='whitegrid', palette='muted')

def create_sequences(X,y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:(i+time_steps)].values)
        ys.append(y.iloc[i:(i+time_steps)])
    return np.array(Xs), np.array(ys)

def setScaler(df:pd.DataFrame, dev_id=None, save=False):

    scaler = StandardScaler()
    _scaler = dict()
    _scaler['L1']  = scaler.fit(df[['L1']])
    _scaler['Tamb']  = scaler.fit(df[['Tamb']])
    _scaler['Tsuc']  = scaler.fit(df[['Tsuc']])
    _scaler['Tliq']  = scaler.fit(df[['Tliq']])
    _scaler['Psuc']  = scaler.fit(df[['Psuc']])
    _scaler['Tsh'] = scaler.fit(df[['Tsh']])
    _scaler['Pliq'] = scaler.fit(df[['Pliq']])
    if(save and dev_id is not None):
        p.dump(_scaler, open(f'models/{dev_id}/{dev_id}_scaler.p', 'wb'))
    return _scaler

def standarize_data(scaler, df:pd.DataFrame, data_steps:int):
    #L1,Tamb,Tsuc,Tliq,Psuc,Tsh,Pliq
    #print(df)
    df['L1'] = scaler['L1'].transform(df[['L1']])
    df['Tamb'] = scaler['Tamb'].transform(df[['Tamb']])
    df['Tsuc'] = scaler['Tsuc'].transform(df[['Tsuc']])
    df['Tliq'] = scaler['Tliq'].transform(df[['Tliq']])
    df['Psuc'] = scaler['Psuc'].transform(df[['Psuc']])
    df['Tsh'] = scaler['Tsh'].transform(df[['Tsh']])
    df['Pliq'] = scaler['Pliq'].transform(df[['Pliq']])
    
    x_df, y_df = create_sequences(df[['L1', 'Tamb', 'Tliq', 'Tsuc', 'Psuc', 'Tsh', 'Pliq']], 
                                    df[['L1', 'Tamb', 'Tliq', 'Tsuc', 'Psuc', 'Tsh', 'Pliq']], 
                                    data_steps)
    
    timesteps = x_df.shape[1]
    num_features = x_df.shape[2]
    
    return x_df, y_df, timesteps, num_features

def get_error(df, prediction, threshold):
    keys = ['L1', 'Tamb', 'Tsuc', 'Tliq', 'Psuc', 'Tsh', 'Pliq']
    p_data = dict()
    r_data = dict()
    anomaly = dict()
    results = dict()

    for key in keys:
        p_data[key] = list()
        r_data[key] = list()

    for data in prediction:
        for d in data:
            p_data['L1'].append(d[0])
            p_data['Tamb'].append(d[1])
            p_data['Tsuc'].append(d[2])
            p_data['Tliq'].append(d[3])
            p_data['Psuc'].append(d[4])
            p_data['Tsh'].append(d[5])
            p_data['Pliq'].append(d[6])
            
    for data in df:
        for d in data:
            r_data['L1'].append(d[0])
            r_data['Tamb'].append(d[1])
            r_data['Tsuc'].append(d[2])
            r_data['Tliq'].append(d[3])
            r_data['Psuc'].append(d[4])
            r_data['Tsh'].append(d[5])
            r_data['Pliq'].append(d[6])

    for key in keys:
        error_array = np.abs(np.array(p_data[key]) - np.array(r_data[key]))
        anomaly[key] = error_array > float(threshold[key])
        results[key] = [len(anomaly[key])-sum(anomaly[key]), # Sem anomalia
                        sum(anomaly[key]), # Anomalia
                        100*(len(anomaly[key])-sum(anomaly[key]))/len(anomaly[key]) # % Anomalia
        ]
    
    return results, anomaly

def error_evaluator(error_dict):
    output_dict = dict()
    for key in error_dict:
        try:
            if float(error_dict[key]) < 20:
                output_dict[key] = 'Normal'
            elif float(error_dict[key]) < 40:
                output_dict[key] = 'Desvio leve'
            elif float(error_dict[key]) < 60:
                output_dict[key] = 'Desvio moderado'
            else:
                output_dict[key] = 'Falha detectada'
        except Exception as e:
            import traceback
            traceback.print_exc()
            output_dict[key] = error_dict[key]

    output_dict['dev_id'] = output_dict['dev_id'][0]
    output_dict['date'] = output_dict['date'].isoformat()
    print(output_dict)
    return output_dict

def plot_var(dev_id, index, df, prediction):
    keys = ['L1', 'Tamb', 'Tsuc', 'Tliq', 'Psuc', 'Tsh', 'Pliq']
    p_data = dict()
    r_data = dict()
    anomaly = dict()
    results = dict()
    for key in keys:
        p_data[key] = list()
        r_data[key] = list()
    i = 0
    for data in prediction:
        for d in data:
            if (i%5 == 0):
                p_data['L1'].append(d[0])
                p_data['Tamb'].append(d[1])
                p_data['Tsuc'].append(d[2])
                p_data['Tliq'].append(d[3])
                p_data['Psuc'].append(d[4])
                p_data['Tsh'].append(d[5])
                p_data['Pliq'].append(d[6])
            i += 1

    i = 0
    for data in df:
        for d in data:
            if (i%5 == 0):
                r_data['L1'].append(d[0])
                r_data['Tamb'].append(d[1])
                r_data['Tsuc'].append(d[2])
                r_data['Tliq'].append(d[3])
                r_data['Psuc'].append(d[4])
                r_data['Tsh'].append(d[5])
                r_data['Pliq'].append(d[6])
            i += 1
    for key in keys:
        error_array = np.abs(np.array(p_data[key]) - np.array(r_data[key]))
        plt.plot(index, r_data[key], label = 'Real')
        plt.plot(index, p_data[key], label = 'Previsto')
        plt.title(f'{dev_id} - {key}')
        plt.legend(loc='best')
        plt.savefig(f'images/{dev_id}_{key}.png')
        plt.clf()

class IntelAutoencoder():
    def __init__(self, dac_id:str , df_train:pd.DataFrame, train_size:float, data_steps=5):
        #Dados e parâmetros de treinamento/partição
        self.dac_id = dac_id
        self.data_steps = data_steps
        self.scaler = setScaler(df_train, dac_id, True)
        self.x_train, self.y_train , timesteps, num_features = standarize_data(self.scaler, df_train, data_steps)
        
        #modelo rede neural
        self.model = Sequential([
            LSTM(128, input_shape=(timesteps, num_features)),
            Dropout(0.2),
            RepeatVector(timesteps),
            LSTM(128, return_sequences=True),
            Dropout(0.2),
            TimeDistributed(Dense(num_features))
        ])
        
    def train_model(self, verbose=True):
        print("Starting...")
        if(verbose):
            v = 1
        else:
            v = 0

        save_path = f"models/{self.dac_id}"

        self.model.compile(loss='mae', optimizer='adam')
        self.model.summary()
            
        self.history = self.model.fit(
            self.x_train, self.y_train,
            epochs = 200,
            batch_size=8,
            validation_split = 0.1,
            #callbacks = [cp_callback],
            shuffle = False,
            verbose=v
        )
        print("Evaluation:", self.model.evaluate(self.x_train, self.y_train))
        self.model.save(save_path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Script para criação de modelo IntelAutoencoder.")
    parser.add_argument('-d',action='append', dest='dev_id', type=str, help='DEV_ID da máquina.')
    args = parser.parse_args()

    sampling = '10Min'
    dev_id = args.dev_id[0]
    print(f"Tratando dados para modelo {dev_id}...")
    df = pd.read_csv(f'training data/{dev_id}.csv', parse_dates=['timestamp'])

    df_treated_list = list()

    df.set_index('timestamp', inplace=True)
    df = df.resample(sampling).max()
    df.dropna(inplace=True)
    
    print(f"Treinando modelo {dev_id}...")
    model= IntelAutoencoder(dac_id= dev_id, df_train = df, train_size = 0.8)
    model.train_model()

