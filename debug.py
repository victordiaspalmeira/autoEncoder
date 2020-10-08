import os, time, datetime
from autoencoder import IntelAutoencoder, get_error, plot_var, error_evaluator
import tensorflow as tf
from tensorflow import keras
from intel_info_connect import insert_error_info
import time, threading
import pandas as pd
import pickle as p
from query_intel import query
import requests as req
import os
import traceback
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
#logging.getLogger().setLevel(logging.DEBUG)

if not os.path.exists("images"):
    os.mkdir("images")

from autoencoder import standarize_data
from memory_profiler import profile 

def request_url(dash_data):
    AUTH = 'xocopilusicosuf' #chave do backend
    url = f'https://api.dielenergia.com/dac/fault-detected?'
    for key in dash_data:
        url = url + f'&{key}={dash_data[key]}'

    url = url + f'&auth={AUTH}'
    resp = req.get(url)
    print(resp.text)

def load_model(dev_id):
    model_path = f"models/{dev_id}"
    return tf.keras.models.load_model(model_path)

def load_scaler(dev_id):
    try:
        file = open(f'models/{dev_id}/{dev_id}_scaler.p', "rb")
        return p.load(open(f'models/{dev_id}/{dev_id}_scaler.p', "rb"))
    except Exception as e:
        print(f'Erro ao recuperar scaler:', e)
        traceback.print_exc()
        return None

def get_daily_data(dev_id, scaler):
    try:
        df = query(dev_id = dev_id, last_hours = 24)
        df['Tsh'] = df['Tsh'].fillna(0)
        df.dropna(inplace=True)
        index = df['Tamb']
        df = standarize_data(scaler, df, 5)
        index = index[:len(df[0])].index 
        return df[0], index
    except Exception as e:
        print(f'Erro ao carregar dados diários de {dev_id}:', e)
        traceback.print_exc()
        return None

def main(dev_id, model, scaler):
    #### dados estáticos ####
    #1. Modelo
    #2. Scaler
    #
    # Avaliar se é mais custo computacional fazer a query a cada iteração ou só manter na memória.

    #### dados dinâmicos ####
    #1. Dados das últimas 24h
    #2. Output
    try:
        threshold = p.load(open(f'models/{dev_id}/{dev_id}_threshold.p', "rb"))
    except Exception as e:
        print(f'{dev_id} threshold fail:', e)
        traceback.print_exc()
        return
    try:
        df, index = get_daily_data(dev_id, scaler)
    except Exception as e:
        print(f'{dev_id} query error:', e)
        traceback.print_exc()
        return
        pass
    try:
        prediction = model.predict_on_batch(df)  
    except Exception as e:
        print(f'{dev_id} Erro ao fazer previsão:', e)
        traceback.print_exc()
        pass

    output, anomaly = get_error(df, prediction, threshold)
    """
    try:
        for k in anomaly['Tsuc']:
            print(k)
    except:
        pass
    """

    data = {
        'dev_id': [dev_id],
        'date': datetime.datetime.now(),
    }
    for key in output:
        data[f'{key}'] = round(100 - output[key][2], 2)
    dash_data = error_evaluator(data)
    #L1, Tamb, Tliq, Tsuc, Psuc, Pliq, Tsh
    
    try:
        insert_error_info(data)
        request_url(dash_data)

    except Exception as e:
        print(f'{dev_id} write error:', e)
    return

if __name__ == "__main__":
    import gc
    dev_list = ['DAC210191015', 'DAC210191016', 
                'DAC210191063', 'DAC210191058', 
                'DAC210191053', 'DAC210191051', 
                'DAC202200053', 'DAC202200010']

    dev_dict = dict()
    for dev in dev_list:
        dev_dict[dev] = (load_model(dev), load_scaler(dev))

    #Máquina 49 usa modelo da 53
    dev_dict['DAC202200049'] = dev_dict['DAC202200053']
    #logging.basicConfig(filename=f'intel logs/periodic.log', format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    while(True):
        for dev in dev_dict:
            main(dev, dev_dict[dev][0], dev_dict[dev][1])
        #main(dev_list[0], model_list[0], scaler_list[0])
        #main(dev_list[1], model_list[1], scaler_list[1])

        collected = gc.collect()
        print("Garbage collector: collected", 
                "%d objects." % collected) 
        time.sleep(3600)