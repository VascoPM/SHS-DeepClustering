# This is a LSTM Auto Enconder model based on the inputs from:
# https://towardsdatascience.com/step-by-step-understanding-lstm-autoencoder-layers-ffab055b6352
# Understanding Deep Learning (Chapter 7.6), Chitta R., 2021

from tensorflow import keras
from keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, RepeatVector, TimeDistributed
from networks import ann_train

def sweep_config(name, window_len, latent_layer_size):
    
    sweep_config = {
    'method': 'random',
    'name': name,
    }
    
    metric = {
    'name': 'mse',
    'goal': 'minimize'
    }
    sweep_config['metric'] = metric
    
    parameters_dict = {
        'optimizer': {
            'values': ['nadam', 'sgd']
        },
        'latent_layer_size': {
            'value': latent_layer_size 
        },
        'epochs':{
            'value': 100
        },
        'window_length':{
            'value': window_len
        },
        'activation_fn':{
            'values': ['SELU','LeakyReLU']
        },
        'learning_rate': {
            # a flat distribution between 0 and 0.1
            'distribution': 'log_uniform_values',
            'min': 0.0001,
            'max': 0.01,
          },
        'batch_size': {
            # integers between 2 and 256
            # with evenly-distributed logarithms 
            'distribution': 'q_log_uniform_values',
            'q': 2,
            'min': 100,
            'max': 300,
          }
        }
    
    sweep_config['parameters'] = parameters_dict

    
  
    return sweep_config


def model(window_length = 90, latent_layer_size = 25, activation_fn = 'SELU'):
    
    inputs = Input(shape= (window_length, 1))
    
    # Encoder
    lstm_e1 = LSTM(100, activation = ann_train.get_activation_fn(activation_fn), return_sequences = True, name='Encoder')(inputs)
    embeding = LSTM(latent_layer_size, activation = ann_train.get_activation_fn(activation_fn), return_sequences = False, name='Lantent_Space')(lstm_e1)
    
    # Decoder
    repeatV = RepeatVector(window_length, name='Reshape_Embeding')(embeding)
    lstm_d1 = LSTM(latent_layer_size, activation = ann_train.get_activation_fn(activation_fn), return_sequences = True, name='Decoder_1')(repeatV)
    lstm_d2 = LSTM(100, activation = ann_train.get_activation_fn(activation_fn), return_sequences = True, name='Decoder_2')(lstm_d1)
    
    outputs = TimeDistributed(Dense(1), name='Reshape_Output')(lstm_d2)
    
    # Full Auto Encoder Model
    autoencoder = keras.models.Model(inputs=inputs, outputs = outputs, name = 'LSTM_AE')
    
    return autoencoder   