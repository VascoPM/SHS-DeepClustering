from tensorflow import keras
from tensorflow.keras.layers import Dense, Input
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
            'min': 0.00001, #v_1 = 0.001
            'max': 0.01, #v_1 = 0.1
        },
        'batch_size': {
            # integers between 2 and 256
            # with evenly-distributed logarithms 
            'distribution': 'q_log_uniform_values',
            'q': 2,
            'min': 100, #v_1 = 2
            'max': 300,
        }
    }
    sweep_config['parameters'] = parameters_dict

   
    return sweep_config

def model(window_length, latent_layer_size, activation_fn):
    
    inputs = Input(shape= window_length)
    
    layer_e1 = Dense(500, activation = ann_train.get_activation_fn(activation_fn),
                     kernel_initializer = ann_train.get_initialization(activation_fn))(inputs) 
    
    layer_e2 = Dense(500, activation = ann_train.get_activation_fn(activation_fn),
                     kernel_initializer = ann_train.get_initialization(activation_fn))(layer_e1) 
    
    layer_e3 = Dense(2000, activation = ann_train.get_activation_fn(activation_fn),
                     kernel_initializer = ann_train.get_initialization(activation_fn))(layer_e2)    
    #Latent Space (no activation)
    encoded = Dense(latent_layer_size)(layer_e3)
       
    layer_d1 = Dense(2000, activation = ann_train.get_activation_fn(activation_fn),
                     kernel_initializer = ann_train.get_initialization(activation_fn))(encoded)
    
    layer_d2 = Dense(500, activation = ann_train.get_activation_fn(activation_fn),
                     kernel_initializer = ann_train.get_initialization(activation_fn))(layer_d1)
    
    layer_d3 = Dense(500, activation = ann_train.get_activation_fn(activation_fn),
                     kernel_initializer = ann_train.get_initialization(activation_fn))(layer_d2)
    # Output Layer, no activation function
    decoded = Dense(window_length)(layer_d3)
       
    autoencoder = keras.models.Model(inputs=inputs, outputs = decoded)
    
    return autoencoder   