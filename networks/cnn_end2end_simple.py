# This CNN Auto Enconder model is based on the Simple CNN Model (fcnn) from:
# https://github.com/blafabregue/TimeSeriesDeepClustering/blob/208fac0343d281f2c5997609916004875aae86fd/networks/fcnn_ae.py
# Encoder filters[64, 64, 64], kernels [3, 5, 9]
# Layer block, CNN - BatchNorm - Activation

from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, Conv1D, Reshape, Activation, BatchNormalization, GlobalAveragePooling1D
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
            'min': 0.00001,
            'max': 0.001,
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
    # CNN Enconder Block 1
    layer_e1 = Conv1D(filters=64, kernel_size=3, padding='same', strides=1)(inputs)
    layer_e1 = BatchNormalization()(layer_e1)
    layer_e1 = Activation(ann_train.get_activation_fn(activation_fn))(layer_e1)
    # CNN Enconder Block 2
    layer_e2 = Conv1D(filters=64, kernel_size=5, padding='same', strides=1)(layer_e1)
    layer_e2 = BatchNormalization()(layer_e2)
    layer_e2 = Activation(ann_train.get_activation_fn(activation_fn))(layer_e2)
    # CNN Enconder Block 3
    layer_e3 = Conv1D(filters=64, kernel_size=9, padding='same', strides=1)(layer_e2)
    layer_e3 = BatchNormalization()(layer_e3)
    layer_e3 = Activation(ann_train.get_activation_fn(activation_fn))(layer_e3)
    
    encoded = GlobalAveragePooling1D()(layer_e3)
    #Latent Space (no activation)
    encoded = Dense(latent_layer_size)(encoded)
    
    # Deconder (Mirror Encoder)
    # Reshaping
    reshape = Dense(window_length)(encoded) # Becasue this architecture doens't have a convolutional upscalling layer (e.g., ConvTrans) a Dense layer is used for this effect (SUB-OPTIMAL)
    reshape = Reshape((window_length, 1))(reshape)
    # CNN Decoder Block 1
    layer_d1 = Conv1D(filters=64, kernel_size=9, padding='same', strides=1)(reshape)
    layer_d1 = Activation(ann_train.get_activation_fn(activation_fn))(layer_d1)    
    # CNN Decoder Block 2
    layer_d2 = Conv1D(filters=64, kernel_size=5, padding='same', strides=1)(layer_d1)
    layer_d2 = Activation(ann_train.get_activation_fn(activation_fn))(layer_d2)    
    # CNN Decoder Block 3
    layer_d3 = Conv1D(filters=64, kernel_size=3, padding='same', strides=1)(layer_d2)
    layer_d3 = Activation(ann_train.get_activation_fn(activation_fn))(layer_d3)    
    # Output Layer
    decoded = Conv1D(filters= 1, kernel_size=9, padding='same', strides=1)(layer_d3)
    
    # Full Auto Encoder Model
    autoencoder = keras.models.Model(inputs=inputs, outputs = decoded, name = 'CNN_E2E')
    
    return autoencoder   