# This CNN Auto Enconder model is based on:
# https://xifengguo.github.io/papers/ICONIP17-DCEC.pdf
# https://github.com/XifengGuo/DCEC

# However, this model does not use DCEC central feature of combining Clustering Loss and Reconstruction Loss
# It only adaptes the Auto Encoder architecture.

# Main Characteristics:
# No pooling Layers, stride instead.
# Fully Connected (FC) Latent Space layer
# Decoder starts with FC layer

# Adaptations:
# This architecture was designed for image data (2D).
# Therefore the size and shape of the layers has to be adapted to a 1D format and 90 day lenght windows


from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, Conv1D, Conv1DTranspose, Activation, BatchNormalization, Flatten, Reshape
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
    # filters=[32, 64, 128], kernel_size = [5,5,3]
    # stride 3 (multiple of 90)
    
    inputs = Input(shape= (window_length, 1))
    # CNN Enconder 
    convL1_e = Conv1D(filters=32, kernel_size=5, padding='same', strides=1)(inputs)
    convL2_e = Conv1D(filters=64, kernel_size=5, padding='same', strides=3)(convL1_e)
    convL3_e = Conv1D(filters=128, kernel_size=3, padding='same', strides=3)(convL2_e)
    
    # Embedding Layer
    flattend = Flatten()(convL3_e)
    encoder = Dense(latent_layer_size)(flattend)
    
    # CNN Decoder
    reshape_dense = Dense(flattend.shape[-1])(encoder)
    reshape_conv = Reshape((10, 128))(reshape_dense)
    
    deconvL1_d = Conv1DTranspose(filters=64, kernel_size=3, padding='same', strides=3)(reshape_conv)
    deconvL2_d = Conv1DTranspose(filters=32, kernel_size=5, padding='same', strides=3)(deconvL1_d)
    decoder_conv = Conv1DTranspose(filters=1, kernel_size=5, padding='same', strides=1)(deconvL2_d)
        
    # Full Auto Encoder Model
    autoencoder = keras.models.Model(inputs=inputs, outputs = decoder_conv, name = 'CNN_DCEC')
    
    return autoencoder   