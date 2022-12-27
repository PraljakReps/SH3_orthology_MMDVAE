

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2DTranspose, Layer, Lambda, Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, BatchNormalization, Activation, Reshape

from keras import backend as K
from keras.constraints import unit_norm, max_norm
from keras.losses import mse, binary_crossentropy
from keras.callbacks import Callback
from keras.layers.advanced_activations import LeakyReLU




def create_encoder(seq_len, aa_var, z_dim, intermediate_dim, alpha = 0.1):
    
    # Encoder comp ...
    encoder_input = Input(shape = (seq_len, aa_var))

    # flatten tensor into a vector
    x = Flatten()(encoder_input)
    
    x = Dense(intermediate_dim * 1.5, kernel_initializer = 'random_normal', )(x)
    x = LeakyReLU(alpha)(x)
    x = Dropout(0.3)(x)
    x = Dense(intermediate_dim * 1.5, kernel_initializer = 'random_normal', )(x)
    x = LeakyReLU(alpha)(x)
    x = Dense(intermediate_dim * 1.5, kernel_initializer = 'random_normal', )(x)
    x = LeakyReLU(alpha)(x)
    
    encoder_output = Dense(z_dim)(x)
    
    return Model(encoder_input, encoder_output, name = 'encoder'), encoder_input


def create_decoder(seq_len, aa_var, z_dim, intermediate_dim, alpha = 0.):

    decoder_input = Input(shape=(z_dim,), name='z_sampling')

    x = Dense(intermediate_dim*1.5, kernel_initializer='random_normal', )(decoder_input)
    x = LeakyReLU(alpha)(x)
    x = Dense(intermediate_dim*1.5, kernel_initializer='random_normal',)(x)
    x = LeakyReLU(alpha)(x)
    x = Dropout(0.7)(x)
    x = Dense(intermediate_dim*1.5,  kernel_initializer='random_normal',)(x)    
    x = LeakyReLU(alpha)(x)

    outputs = Dense(intermediate_dim, activation='linear', kernel_initializer='random_normal',)(x)                 
    outputs = Reshape((seq_len, aa_var))(outputs)
    softmax = tf.keras.activations.softmax(outputs, axis=-1)
    
    return Model(decoder_input, softmax, name = 'decoder') 


#MMD loss
def compute_kernel(x, y):

    x_size = K.shape(x)[0]
    y_size = K.shape(y)[0]
    dim = K.shape(x)[1]
    tiled_x = K.tile(K.reshape(x, [x_size, 1, dim]), [1, y_size, 1])
    tiled_y = K.tile(K.reshape(y, [1, y_size, dim]), [x_size, 1, 1])
    return K.exp(-K.mean(K.square(tiled_x - tiled_y), axis = 2) / K.cast(dim, 'float32'))

def compute_mmd(x, y):
    x_kernel = compute_kernel(x,x)
    y_kernel = compute_kernel(y,y)
    xy_kernel = compute_kernel(x,y)
    return K.mean(x_kernel) + K.mean(y_kernel) - 2 * K.mean(xy_kernel)

def mmdVAE_loss(train_z, train_xr, train_x):

    # sampling from the random noise ...
    batch_size = K.shape(train_z)[0]
    latent_dim = K.int_shape(train_z)[1]
    true_samples = K.random_normal(shape = (batch_size, latent_dim), mean = 0., stddev = 1.)

    # calc MMD loss
    loss_MMD = compute_mmd(true_samples, train_z)

    # calc the reconstruction loss (i.e. negative log-likelihood)
    loss_REC = K.mean(K.square(train_xr - train_x))

    return loss_REC + 2*loss_MMD





