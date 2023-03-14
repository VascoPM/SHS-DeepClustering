# This CNN Auto Enconder model is based on suggestions from books:
# Understanding Deep Learning (Chapter:7.6.2 Convolutional Autoencoder)
# Hands-on-Machine-Learning 2nd Ed


# WHAT TO TRY:
# Make a CNN architecture with Conv blocks (Conv, BN, Maxpool), based on Books' advice
# Make a CNN with Conv1D only, with strides like: https://xifengguo.github.io/papers/ICONIP17-DCEC.pdf
# Make a CNN with a Conv embeded space: https://arxiv.org/pdf/1805.10795v1.pdf

from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, Conv1D, Conv1DTranspose, Activation, BatchNormalization, MaxPool1D, Flatten, Reshape
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

def model(window_length, latent_layer_size, activation_fn = 'SELU'):
    # Hands-on-ML inputs:
    # kernel size: small kernels better (e.g., 3), maybe first layer larger kernel ok since there's few input channels
    # N filters: Increase deeper it goes, as there are more patterns the higher the abstraction (DOUBLECHECK)
    
    # Understanding Deep Learning:
    # (Decoder) Don't used Conv1D or Pool layers: these layers summarize data instead of reconstructing it (upscaling), therefore should use Conv1DTranspose instead
    # (Decoder) BatchNorm: should help the decoder with exploding gradients 
    
    inputs = Input(shape= (window_length, 1))
    # CNN Enconder Block 1
    convB1_e = Conv1D(filters=32, kernel_size=5, padding='same', strides=1)(inputs)
    convB1_e = BatchNormalization()(convB1_e)
    convB1_e = Activation(ann_train.get_activation_fn(activation_fn))(convB1_e)
    convB1_e = MaxPool1D(pool_size = 3)(convB1_e)
    # CNN Enconder Block 2
    convB2_e = Conv1D(filters=64, kernel_size=3, padding='same', strides=1)(convB1_e)
    convB2_e = BatchNormalization()(convB2_e)
    convB2_e = Activation(ann_train.get_activation_fn(activation_fn))(convB2_e)
    convB2_e = MaxPool1D(pool_size = 3)(convB2_e)
    
    #Latent Space (no activation)
    flattend = Flatten()(convB2_e)
    encoded = Dense(latent_layer_size)(flattend)
    
    # Deconder (Mirror Encoder)
    # Reshaping
    # check: https://xifengguo.github.io/papers/ICONIP17-DCEC.pdf
    reshape = Dense(flattend.shape[-1])(encoded)
    reshape = Reshape((10, 64))(reshape)
    # CNN Decoder Block 1
    convB1_d = Conv1DTranspose(filters=64, kernel_size=3, padding='same', strides=3)(reshape)
    convB1_d = BatchNormalization()(convB1_d)
    convB1_d = Activation(ann_train.get_activation_fn(activation_fn))(convB1_d)    
    # CNN Decoder Block 1
    convB2_d = Conv1DTranspose(filters=32, kernel_size=5, padding='same', strides=3)(convB1_d)
    convB2_d = BatchNormalization()(convB2_d)
    convB2_d = Activation(ann_train.get_activation_fn(activation_fn))(convB2_d)    
    # Output Layer
    conv_decoded = Conv1DTranspose(filters=1, kernel_size=3, padding='same', strides=1)(convB2_d)
    flattend_decoded = Flatten()(conv_decoded)
    decoded = Dense(window_length)(flattend_decoded)
    # ERROR, Output shape != Input shape
    
    # Full Auto Encoder Model
    autoencoder = keras.models.Model(inputs=inputs, outputs = decoded)
    
    return autoencoder   