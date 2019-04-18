import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import gc

from keras.models import Sequential
from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam
from utils import *
import keras.backend as K
from glob import glob


def define_unet_4(verbose=False):
    inputs = Input((320, 256, 1))
    conv1 = Convolution2D(8, 3, padding = 'same', activation='relu', kernel_initializer='he_normal')(inputs)
    pool1 = MaxPooling2D(padding='same')(conv1)
    conv2 = Convolution2D(8, 3, padding = 'same', activation='relu', kernel_initializer='he_normal')(pool1)
    conv3 = Convolution2D(16, 3, padding = 'same', activation='relu', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(padding='same')(conv3)
    conv4 = Convolution2D(16, 3, padding = 'same', activation='relu', kernel_initializer='he_normal')(pool2)
    conv5 = Convolution2D(32, 3, padding = 'same', activation='relu', kernel_initializer='he_normal')(conv4)
    pool3 = MaxPooling2D(padding='same')(conv5)
    conv6 = Convolution2D(32, 3, padding = 'same', activation='relu', kernel_initializer='he_normal')(pool3)
    conv7 = Convolution2D(64, 3, padding = 'same', activation='relu', kernel_initializer='he_normal')(conv6)
    pool3 = MaxPooling2D(padding='same')(conv7)
    conv8 = Convolution2D(64, 3, padding = 'same', activation='relu', kernel_initializer='he_normal')(pool3)
    flat1 = Flatten()(conv8)
    dense1 = Dense(((320 // (2**4)) * (256 // (2**4))), activation = 'tanh')(flat1)
    dense1 = Reshape(((320 // (2**4)), (256 // (2**4)), 1))(dense1)
    conv9 = Convolution2D(128, 3, padding = 'same', activation='relu', kernel_initializer='he_normal')(dense1)
    conv10 = Convolution2D(128, 1, padding = 'same', activation='relu', kernel_initializer='he_normal')(conv9)
    ups1 = UpSampling2D()(conv10)
    conv11 = Convolution2D(64, 3, padding = 'same', activation='relu', kernel_initializer='he_normal')(ups1)
    merge1 = concatenate([conv11, conv7], axis = 3)
    conv12 = Convolution2D(64, 1, padding = 'same', activation='relu', kernel_initializer='he_normal')(merge1)
    ups2 = UpSampling2D()(conv12)
    conv13 = Convolution2D(32, 3, padding = 'same', activation='relu', kernel_initializer='he_normal')(ups2)
    merge2 = concatenate([conv13, conv5], axis = 3)
    conv14 = Convolution2D(32, 1, padding = 'same', activation='relu', kernel_initializer='he_normal')(merge2)
    ups3 = UpSampling2D()(conv14)
    conv15 = Convolution2D(16, 3, padding = 'same', activation='relu', kernel_initializer='he_normal')(ups3)
    merge3 = concatenate([conv15, conv3], axis = 3)
    conv16 = Convolution2D(16, 1, padding = 'same', activation='relu', kernel_initializer='he_normal')(merge3)
    ups4 = UpSampling2D()(conv16)
    conv17 = Convolution2D(1, 3, padding = 'same', activation='tanh', kernel_initializer='he_normal')(ups4)

    model = Model(input=inputs, output=conv17)
    if verbose:
        model.summary()
    return model

# 5 upsampling, downsampling layers

def define_network_5(verbose=False):
    autoencoder = Sequential([
        Convolution2D(8, 3, padding = 'same', input_shape = (320, 256, 1)), 
        MaxPooling2D(padding='same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1),
        Activation('relu'), 
        Convolution2D(8, 3, padding = 'same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'), 
        Activation('relu'), 
        Convolution2D(16, 3, padding = 'same'),
        MaxPooling2D(padding='same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'), 
        Activation('relu'), 
        Convolution2D(16, 3, padding = 'same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        Activation('relu'), 
        Convolution2D(32, 3, padding = 'same'),
        MaxPooling2D(padding='same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        Activation('relu'), 
        Convolution2D(32, 3, padding = 'same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        Activation('relu'), 
        Convolution2D(64, 3, padding = 'same'),   
        MaxPooling2D(padding='same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        Activation('relu'), 
        Convolution2D(64, 3, padding = 'same'),   
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        Activation('relu'), 
        
        Convolution2D(128, 3, padding = 'same'),
        MaxPooling2D(padding='same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        Activation('relu'), 
        Convolution2D(128, 3, padding = 'same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        Activation('relu'), 
        ####
        Flatten(),
        Dense(((320 // (2**5)) * (256 // (2**5))), activation = 'tanh'),
        # Dense(192, input_shape = (encoding_size,), activation = 'relu'),
        Reshape(((320 // (2**5)), (256 // (2**5)), 1)),
        #BatchNormalization(),
        Convolution2D(256, 3, padding = 'same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        Activation('relu'), 
        Convolution2D(256, 1, padding = 'same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        Activation('relu'), 
        UpSampling2D(),
        Convolution2D(128, 3, padding = 'same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        Activation('relu'), 
        Convolution2D(128, 1, padding = 'same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        Activation('relu'), 
        UpSampling2D(),
        #BatchNormalization(),
        Convolution2D(64, 3, padding = 'same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        Activation('relu'), 
        Convolution2D(64, 1, padding = 'same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        Activation('relu'), 
        UpSampling2D(),
        #BatchNormalization(),
        Convolution2D(32, 3, padding = 'same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        Activation('relu'), 
        Convolution2D(32, 1, padding = 'same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        Activation('relu'), 
        UpSampling2D(),
        #BatchNormalization(),
        Convolution2D(16, 3, padding = 'same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        Activation('relu'), 
        Convolution2D(16, 1, padding = 'same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        Activation('relu'), 
        UpSampling2D(),
        #BatchNormalization(),
        Convolution2D(1, 3, padding = 'same'),
        #BatchNormalization(),
        Activation('tanh')
    ])
    if verbose:
        print(autoencoder.summary())
    return autoencoder

# 4 upsampling, downsampling layers

def define_network_4(verbose=False):
    autoencoder = Sequential([
        Convolution2D(8, 3, padding = 'same', input_shape = (320, 256, 1)), 
        MaxPooling2D(padding='same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1),
        Activation('relu'), 
        Convolution2D(8, 3, padding = 'same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'), 
        Activation('relu'), 
        Convolution2D(16, 3, padding = 'same'),
        MaxPooling2D(padding='same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'), 
        Activation('relu'), 
        Convolution2D(16, 3, padding = 'same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        Activation('relu'), 
        Convolution2D(32, 3, padding = 'same'),
        MaxPooling2D(padding='same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        Activation('relu'), 
        Convolution2D(32, 3, padding = 'same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        Activation('relu'), 
        Convolution2D(64, 3, padding = 'same'),   
        MaxPooling2D(padding='same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        Activation('relu'), 
        Convolution2D(64, 3, padding = 'same'),   
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        Activation('relu'), 
        ####
        ##Convolution2D(128, 3, padding = 'same'),
        ##MaxPooling2D(padding='same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        ##Activation('relu'), 
        ##Convolution2D(128, 3, padding = 'same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        ##Activation('relu'), 
        ####
        Flatten(),
        Dense(((320 // (2**4)) * (256 // (2**4))), activation = 'tanh'),
        # Dense(192, input_shape = (encoding_size,), activation = 'relu'),
        Reshape(((320 // (2**4)), (256 // (2**4)), 1)),
        #BatchNormalization(),
        Convolution2D(128, 3, padding = 'same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        Activation('relu'), 
        Convolution2D(128, 1, padding = 'same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        Activation('relu'), 
        UpSampling2D(),
        #BatchNormalization(),
        Convolution2D(64, 3, padding = 'same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        Activation('relu'), 
        Convolution2D(64, 1, padding = 'same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        Activation('relu'), 
        UpSampling2D(),
        #BatchNormalization(),
        Convolution2D(32, 3, padding = 'same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        Activation('relu'), 
        Convolution2D(32, 1, padding = 'same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        Activation('relu'), 
        UpSampling2D(),
        #BatchNormalization(),
        Convolution2D(16, 3, padding = 'same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        Activation('relu'), 
        Convolution2D(16, 1, padding = 'same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        Activation('relu'), 
        UpSampling2D(),
        #BatchNormalization(),
        Convolution2D(1, 3, padding = 'same'),
        #BatchNormalization(),
        Activation('tanh')
    ])
    if verbose:
        print(autoencoder.summary())
    return autoencoder

# 3 upsampling, downsampling layers

def define_network_3(verbose=False):
    autoencoder = Sequential([
        Convolution2D(8, 3, padding = 'same', input_shape = (320, 256, 1)), 
        MaxPooling2D(padding='same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1),
        Activation('relu'), 
        Convolution2D(8, 3, padding = 'same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'), 
        Activation('relu'), 
        Convolution2D(16, 3, padding = 'same'),
        MaxPooling2D(padding='same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'), 
        Activation('relu'), 
        Convolution2D(16, 3, padding = 'same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        Activation('relu'), 
        Convolution2D(32, 3, padding = 'same'),
        MaxPooling2D(padding='same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        Activation('relu'), 
        Convolution2D(32, 3, padding = 'same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        Activation('relu'), 
        ####
        ##Convolution2D(128, 3, padding = 'same'),
        ##MaxPooling2D(padding='same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        ##Activation('relu'), 
        ##Convolution2D(128, 3, padding = 'same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        ##Activation('relu'), 
        ####
        Flatten(),
        Dense(((320 // (2**3)) * (256 // (2**3))), activation = 'tanh'),
        # Dense(192, input_shape = (encoding_size,), activation = 'relu'),
        Reshape(((320 // (2**3)), (256 // (2**3)), 1)),
        #BatchNormalization(),
        #BatchNormalization(),
        Convolution2D(64, 3, padding = 'same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        Activation('relu'), 
        Convolution2D(64, 1, padding = 'same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        Activation('relu'), 
        UpSampling2D(),
        #BatchNormalization(),
        Convolution2D(32, 3, padding = 'same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        Activation('relu'), 
        Convolution2D(32, 1, padding = 'same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        Activation('relu'), 
        UpSampling2D(),
        #BatchNormalization(),
        Convolution2D(16, 3, padding = 'same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        Activation('relu'), 
        Convolution2D(16, 1, padding = 'same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        Activation('relu'), 
        UpSampling2D(),
        #BatchNormalization(),
        Convolution2D(1, 3, padding = 'same'),
        #BatchNormalization(),
        Activation('tanh')
    ])
    if verbose:
        print(autoencoder.summary())
    return autoencoder

# 2 upsampling, downsampling layers

def define_network_2(verbose=False):
    autoencoder = Sequential([
        Convolution2D(8, 3, padding = 'same', input_shape = (320, 256, 1)), 
        MaxPooling2D(padding='same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1),
        Activation('relu'), 
        Convolution2D(8, 3, padding = 'same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'), 
        Activation('relu'), 
        Convolution2D(16, 3, padding = 'same'),
        MaxPooling2D(padding='same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'), 
        Activation('relu'), 
        Convolution2D(16, 3, padding = 'same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        Activation('relu'), 
        ####
        ##Convolution2D(128, 3, padding = 'same'),
        ##MaxPooling2D(padding='same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        ##Activation('relu'), 
        ##Convolution2D(128, 3, padding = 'same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        ##Activation('relu'), 
        ####
        Flatten(),
        Dense(((320 // (2**2)) * (256 // (2**2))), activation = 'tanh'),
        # Dense(192, input_shape = (encoding_size,), activation = 'relu'),
        Reshape(((320 // (2**2)), (256 // (2**2)), 1)),
        #BatchNormalization(),
        #BatchNormalization(),
        #BatchNormalization(),
        Convolution2D(32, 3, padding = 'same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        Activation('relu'), 
        Convolution2D(32, 1, padding = 'same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        Activation('relu'), 
        UpSampling2D(),
        #BatchNormalization(),
        Convolution2D(16, 3, padding = 'same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        Activation('relu'), 
        Convolution2D(16, 1, padding = 'same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        Activation('relu'), 
        UpSampling2D(),
        #BatchNormalization(),
        Convolution2D(1, 3, padding = 'same'),
        #BatchNormalization(),
        Activation('tanh')
    ])
    if verbose:
        print(autoencoder.summary())
    return autoencoder

# 1 upsampling, downsampling layers
## DONT USE - too many params


def define_network_1(verbose=False):
    autoencoder = Sequential([
        Convolution2D(8, 3, padding = 'same', input_shape = (320, 256, 1)), 
        MaxPooling2D(padding='same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1),
        Activation('relu'), 
        Convolution2D(8, 3, padding = 'same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'), 
        Activation('relu'), 
        ####
        ##Convolution2D(128, 3, padding = 'same'),
        ##MaxPooling2D(padding='same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        ##Activation('relu'), 
        ##Convolution2D(128, 3, padding = 'same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        ##Activation('relu'), 
        ####
        Flatten(),
        Dense(((320 // (2**1)) * (256 // (2**1))), activation = 'tanh'),
        # Dense(192, input_shape = (encoding_size,), activation = 'relu'),
        Reshape(((320 // (2**1)), (256 // (2**1)), 1)),
        #BatchNormalization(),
        #BatchNormalization(),
        #BatchNormalization(),
        #BatchNormalization(),
        Convolution2D(16, 3, padding = 'same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        Activation('relu'), 
        Convolution2D(16, 1, padding = 'same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        Activation('relu'), 
        UpSampling2D(),
        #BatchNormalization(),
        Convolution2D(1, 3, padding = 'same'),
        #BatchNormalization(),
        Activation('tanh')
    ])
    if verbose:
        print(autoencoder.summary())
    return autoencoder

def define_network_4_2chan(verbose=False):
    autoencoder = Sequential([
        Convolution2D(8, 3, padding = 'same', input_shape = (320, 256, 2)), 
        MaxPooling2D(padding='same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1),
        Activation('relu'), 
        Convolution2D(8, 3, padding = 'same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'), 
        Activation('relu'), 
        Convolution2D(16, 3, padding = 'same'),
        MaxPooling2D(padding='same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'), 
        Activation('relu'), 
        Convolution2D(16, 3, padding = 'same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        Activation('relu'), 
        Convolution2D(32, 3, padding = 'same'),
        MaxPooling2D(padding='same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        Activation('relu'), 
        Convolution2D(32, 3, padding = 'same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        Activation('relu'), 
        Convolution2D(64, 3, padding = 'same'),   
        MaxPooling2D(padding='same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        Activation('relu'), 
        Convolution2D(64, 3, padding = 'same'),   
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        Activation('relu'), 
        ####
        ##Convolution2D(128, 3, padding = 'same'),
        ##MaxPooling2D(padding='same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        ##Activation('relu'), 
        ##Convolution2D(128, 3, padding = 'same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        ##Activation('relu'), 
        ####
        Flatten(),
        Dense(((320 // (2**4)) * (256 // (2**4)) * 2), activation = 'tanh'),
        # Dense(192, input_shape = (encoding_size,), activation = 'relu'),
        Reshape(((320 // (2**4)), (256 // (2**4)), 2)),
        #BatchNormalization(),
        Convolution2D(128, 3, padding = 'same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        Activation('relu'), 
        Convolution2D(128, 1, padding = 'same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        Activation('relu'), 
        UpSampling2D(),
        #BatchNormalization(),
        Convolution2D(64, 3, padding = 'same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        Activation('relu'), 
        Convolution2D(64, 1, padding = 'same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        Activation('relu'), 
        UpSampling2D(),
        #BatchNormalization(),
        Convolution2D(32, 3, padding = 'same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        Activation('relu'), 
        Convolution2D(32, 1, padding = 'same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        Activation('relu'), 
        UpSampling2D(),
        #BatchNormalization(),
        Convolution2D(16, 3, padding = 'same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        Activation('relu'), 
        Convolution2D(16, 1, padding = 'same'),
        #BatchNormalization(),
        #LeakyReLU(alpha=0.1), #Activation('relu'),
        Activation('relu'), 
        UpSampling2D(),
        #BatchNormalization(),
        Convolution2D(2, 3, padding = 'same'),
        #BatchNormalization(),
        Activation('tanh')
    ])
    if verbose:
        print(autoencoder.summary())
    return autoencoder