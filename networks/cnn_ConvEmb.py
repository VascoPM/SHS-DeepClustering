# This CNN Auto Enconder model is based on:
# https://www.sciencedirect.com/science/article/pii/S0031320318301936#fig0002
# Later adapted in:
# https://arxiv.org/pdf/1805.10795v1.pdf
# https://github.com/liuyilin950623/Deep_Discriminative_Clustering

# This implementation discards the Clustering Loss procedure and adapts only the Auto Encoder architecture itself.

# Note that no Un-pooling Layer is implemented, as the only repository found does not implement it:
# https://github.com/liuyilin950623/Deep_Discriminative_Clustering/blob/master/model/model.py
# And according to following thred, Unpooling does not have a straight foward implementation in Keras/Tensorflow:
# https://stackoverflow.com/questions/36548736/tensorflow-unpooling

# Main Characteristics:
# No Fully Connected (FC), all CNN layers, including Embeding
# [Note page 487 (Hands-on-ML 2nd ed) shows how to convert a FC to a CNN, thinking of the embeding layer.]
# Down-sampling with MaxPool1D layers, intead of stride
# Upsampling with Conv1DTranspose and stride. (No UNpooling examples found)

# Adaptations:
# This architecture was designed for image data (2D).
# Therefore the size and shape of the layers has to be adapted to a 1D format and 90 day lenght windows


from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, Conv1D, Conv1DTranspose, Activation, BatchNormalization, Reshape, MaxPool1D
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
            'min': 0.001,
            'max': 0.01,
          },
        'batch_size': {
            # integers between 2 and 256
            # with evenly-distributed logarithms 
            'distribution': 'q_log_uniform_values',
            'q': 2,
            'min': 200,
            'max': 300,
          }
        }
    
    sweep_config['parameters'] = parameters_dict

    
    return sweep_config

def model(window_length = 90, latent_layer_size = 25, activation_fn = 'SELU'):
    # filters=[6, 16, 60], kernel_size = [5,3,3] (guess)
    # Note, Down Sampling is also achieved with 'valid' padding and kernel size 
    # But unlike reference this is only used in the Embeding layer to achieve the desired Latent Space dimensions (depth-wise)
    
    
    inputs = Input(shape= (window_length, 1))
    
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
    
    #Conv Embeding layer
    conv_embeding = Conv1D(filters=latent_layer_size, kernel_size=5, padding='valid', strides=1, name = 'Latent_Space')(convB3_e)
    
    # CNN Decoder Block 1
    convB1_d = Conv1DTranspose(filters=60, kernel_size=5, padding='valid', strides=1, name = '1BD_Conv')(conv_embeding)
    convB1_d = BatchNormalization(name = '1BD_BN')(convB1_d)
    convB1_d = Activation(ann_train.get_activation_fn(activation_fn), name = '1BD_Act')(convB1_d)
    # (Upscaling)
    convB1_d = Conv1DTranspose(filters=60, kernel_size=2, padding='same', strides=2, name = '1BD_Upscale')(convB1_d)
    convB1_d = Activation(ann_train.get_activation_fn(activation_fn), name = '1BD_UpAct')(convB1_d)
    
    # CNN Decoder Block 2
    convB2_d = Conv1DTranspose(filters=16, kernel_size=3, padding='same', strides=1, name = '2BD_Conv')(convB1_d)
    convB2_d = BatchNormalization(name = '2BD_BN')(convB2_d)
    convB2_d = Activation(ann_train.get_activation_fn(activation_fn), name = '2BD_Act')(convB2_d)   
    # (Upscaling)
    convB2_d = Conv1DTranspose(filters=16, kernel_size=2, padding='same', strides=3, name = '2BD_Upscale')(convB2_d)
    convB2_d = Activation(ann_train.get_activation_fn(activation_fn), name = '2BD_UpAct')(convB2_d)
    
    # CNN Decoder Block 3
    convB3_d = Conv1DTranspose(filters=6, kernel_size=5, padding='same', strides=1, name = '3BD_Conv')(convB2_d)
    convB3_d = BatchNormalization(name = '3BD_BN')(convB3_d)
    convB3_d = Activation(ann_train.get_activation_fn(activation_fn), name = '3BD_Act')(convB3_d)
    # (Upscaling)
    convB3_d = Conv1DTranspose(filters=6, kernel_size=2, padding='same', strides=3, name = '3BD_Upscale')(convB3_d)
    convB3_d = Activation(ann_train.get_activation_fn(activation_fn), name = '3BD_UpAct')(convB3_d)
    
    # Output
    output = Conv1DTranspose(filters=1, kernel_size=5, padding='same', strides=1, name = 'Output')(convB3_d)
   
        
    # Full Auto Encoder Model
    autoencoder = keras.models.Model(inputs=inputs, outputs = output, name = 'CNN_ConvEmb')
    
    return autoencoder   