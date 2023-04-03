# This CNN Auto Enconder model is based on suggestions from books:
# Understanding Deep Learning (Chapter:7.6.2 Convolutional Autoencoder)
# Hands-on-Machine-Learning 2nd Ed

 # Hands-on-ML inputs:
# kernel size: small kernels better (e.g., 3), maybe first layer larger kernel ok since there's few input channels
# N filters: Increase deeper it goes, as there are more patterns the higher the abstraction 

# Understanding Deep Learning:
# (Decoder) Don't used Conv1D or Pool layers: these layers summarize data instead of reconstructing it (upscaling), therefore should use Conv1DTranspose instead
# (Decoder) BatchNorm: should help the decoder with exploding gradients 



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
            'values': ['nadam']
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
            'values': ['LeakyReLU']
        },
        'learning_rate': {
            # a flat distribution between 0 and 0.1
            'distribution': 'log_uniform_values',
            'min': 0.0001,
            'max': 0.01,
          },
        'lr_decay': {
            'value': 0.01
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
    # Conv Block 1
    convB1_e = Conv1D(filters=6, kernel_size=5, padding='same', strides=1, name = '1BE_Conv')(inputs)
    convB1_e = BatchNormalization(name = '1BE_BN')(convB1_e)
    convB1_e = Activation(ann_train.get_activation_fn(activation_fn), name = '1BE_Act')(convB1_e)
    convB1_e = MaxPool1D(pool_size = 3, name = '1BE_MaxP')(convB1_e)
    # Conv Block 2
    convB2_e = Conv1D(filters=16, kernel_size=3, padding='same', strides=1, name = '2BE_Conv')(convB1_e)
    convB2_e = BatchNormalization(name = '2BE_BN')(convB2_e)
    convB2_e = Activation(ann_train.get_activation_fn(activation_fn), name = '2BE_Act')(convB2_e)
    convB2_e = MaxPool1D(pool_size = 3, name = '2BE_MaxP')(convB2_e)
    # Conv Block 3
    convB3_e = Conv1D(filters=60, kernel_size=3, padding='same', strides=1, name = '3BE_Conv')(convB2_e)
    convB3_e = BatchNormalization(name = '3BE_BN')(convB3_e)
    convB3_e = Activation(ann_train.get_activation_fn(activation_fn), name = '3BE_Act')(convB3_e)
    convB3_e = MaxPool1D(pool_size = 2, name = '3BE_MaxP')(convB3_e)
    # Embedding Layer
    flattend = Flatten(name='Encoder_Reshape')(convB3_e)
    encoder = Dense(latent_layer_size, name='Latent_Space')(flattend)
    # Decoder Reshaping
    reshape_dense = Dense(flattend.shape[-1], name='Decoder_Reshape1')(encoder)
    reshape_conv = Reshape((convB3_e.shape[1], convB3_e.shape[2]), name='Decoder_Reshape2')(reshape_dense)
    # CNN Decoder Block 1
    convB1_d = Conv1DTranspose(filters=60, kernel_size=3, padding='same', strides=2, name = '1BD_Conv')(reshape_conv)
    convB1_d = BatchNormalization(name = '1BD_BN')(convB1_d)
    convB1_d = Activation(ann_train.get_activation_fn(activation_fn), name = '1BD_Act')(convB1_d)
    # CNN Decoder Block 2
    convB2_d = Conv1DTranspose(filters=16, kernel_size=3, padding='same', strides=3, name = '2BD_Conv')(convB1_d)
    convB2_d = BatchNormalization(name = '2BD_BN')(convB2_d)
    convB2_d = Activation(ann_train.get_activation_fn(activation_fn), name = '2BD_Act')(convB2_d)#
    # CNN Decoder Block 2
    convB3_d = Conv1DTranspose(filters=6, kernel_size=5, padding='same', strides=3, name = '3BD_Conv')(convB2_d)
    convB3_d = BatchNormalization(name = '3BD_BN')(convB3_d)
    convB3_d = Activation(ann_train.get_activation_fn(activation_fn), name = '3BD_Act')(convB3_d)
    
    # Output as Conv1D
    decoder = Conv1DTranspose(filters=1, kernel_size=5, padding='same', strides=1, activation='linear', name = 'Output')(convB3_d)    
    
    # Full Auto Encoder Model
    autoencoder = keras.models.Model(inputs=inputs, outputs = decoder, name = 'CNN_AE')
    
    return autoencoder   