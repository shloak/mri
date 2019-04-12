import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import gc

from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Activation, MaxPooling2D, BatchNormalization, LeakyReLU
from keras.layers import Convolution2D
from keras.layers import AveragePooling2D
from keras.layers import UpSampling2D
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Input
from keras.optimizers import Adam
from utils import *
import keras.backend as K
from glob import glob

config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) 
config.gpu_options.allow_growth=True
sess = tf.Session(config=config) 
K.set_session(sess)

data = glob("./data/test_img_slices/*.ra")
minv, maxv = 0, 0
#for i in range(len(data)):
#    img = get_image_old2(data[i])
#    minv = min(minv, np.min(img[:, :, 0]))
#    maxv = max(maxv, np.max(img[:, :, 0]))
#print(minv, maxv)

#img = get_image_old2(data[0]) 
img = get_image_old2('./data/test_img_slices/19_100.ra') # same - for linux
plt.imshow(img[:, :, 0], cmap='gray')
plt.show()
minv = np.min(img[:, :, 0])
maxv = np.max(img[:, :, 0])
#plt.imshow(-1 + (2 * np.array(img[:, :, 0]) / (maxv - minv)), cmap='gray')
normalized_img = np.expand_dims(-1 + (2 * (np.array(img[:, :, 0] - minv) / (maxv - minv))), 2)
plt.imshow(normalized_img[:, :, 0], cmap='gray')
plt.show()
minv = np.min(normalized_img[:, :, 0])
maxv = np.max(normalized_img[:, :, 0])
print(minv, maxv)

#img2 = get_image_old2(data[70])
img2 = get_image_old2('./data/test_img_slices/19_170.ra') # same - for linux
plt.imshow(img2[:, :, 0], cmap='gray')
plt.show()
minv = np.min(img2[:, :, 0])
maxv = np.max(img2[:, :, 0])
#plt.imshow(-1 + (2 * np.array(img[:, :, 0]) / (maxv - minv)), cmap='gray')
normalized_img2 = np.expand_dims(-1 + (2 * (np.array(img2[:, :, 0] - minv) / (maxv - minv))), 2)
plt.imshow(normalized_img2[:, :, 0], cmap='gray')
plt.show()
minv = np.min(normalized_img2[:, :, 0])
maxv = np.max(normalized_img2[:, :, 0])
print(minv, maxv)

mask_files = glob("./masks/gen_masks/2_0*")
mask_files = [np.fft.fftshift(np.load(m)) for m in mask_files]
mask = mask_files[0]
new_img = np.fft.ifft2(np.multiply(np.fft.fft2(normalized_img[:, :, 0]), mask)).astype(float)
plt.imshow(new_img, cmap='gray')
plt.show()
minv = np.min(new_img)
maxv = np.max(new_img)
normalized_new_img = np.expand_dims(-1 + (2 * (new_img - minv) / (maxv - minv)), 2)
plt.imshow(normalized_new_img[:, :, 0], cmap='gray')
plt.show()
print(np.min(normalized_new_img))
print(np.max(normalized_new_img))
image_shape = normalized_img.shape


encoding_size = 16*20 # 128
perturbation_max = 40

preprocess = lambda x : x / 127 - 1
deprocess  = lambda x :((x + 1) * 127).astype(np.uint8)

def get_image_normalized(index, plot=False):
    data = glob("./data/test_img_slices/*.ra")
    img = get_image_old2('./data/test_img_slices/19_{}.ra'.format(index)) # same - for linux
    minv = np.min(img[:, :, 0])
    maxv = np.max(img[:, :, 0])
    normalized_img = np.expand_dims(-1 + (2 * (np.array(img[:, :, 0] - minv) / (maxv - minv))), 2)
    if plot:
        plt.imshow(normalized_img[:, :, 0], cmap='gray')
        plt.show()
    minv = np.min(normalized_img[:, :, 0])
    maxv = np.max(normalized_img[:, :, 0])
    #print(minv, maxv)
    return normalized_img

def get_batch_normalized(indices):
    return np.array([get_image_normalized(ind) for ind in indices])


def get_subsampled_normalized(normalized_img, subs, plot=False):
    mask_files = glob("./masks/gen_masks/{}_0*".format(subs))
    mask_files = [np.fft.fftshift(np.load(m)) for m in mask_files]
    mask = mask_files[0]
    new_img = np.fft.ifft2(np.multiply(np.fft.fft2(normalized_img[:, :, 0]), mask)).astype(float)
    minv = np.min(new_img)
    maxv = np.max(new_img)
    normalized_new_img = np.expand_dims(-1 + (2 * (new_img - minv) / (maxv - minv)), 2)
    if plot:
        plt.imshow(normalized_new_img[:, :, 0], cmap='gray')
        plt.show()
    #print(np.min(normalized_new_img))
    #print(np.max(normalized_new_img))
    return normalized_new_img

def get_image_normalized_2chan(index, plot=False):
    data = glob("./data/test_img_slices/*.ra")
    img = get_image_old2('./data/test_img_slices/19_{}.ra'.format(index)) # same - for linux
    minv = np.min(img[:, :, 0])
    maxv = np.max(img[:, :, 0])
    normalized_img = np.zeros((320, 256, 2))
    normalized_img_real = -1 + (2 * (np.array(img[:, :, 0] - minv) / (maxv - minv)))
    normalized_img[:, :, 0] = normalized_img_real
    normalized_img[:, :, 1] = img[:, :, 1]
    if plot:
        plt.imshow(np.sqrt(normalized_img[:, :, 0] ** 2 + normalized_img[:, :, 1] ** 2), cmap='gray')
        plt.show()
    minv = np.min(normalized_img[:, :, 0])
    maxv = np.max(normalized_img[:, :, 0])
    #print(minv, maxv)
    return normalized_img

def get_batch_normalized_2chan(indices):
    return np.array([get_image_normalized_2chan(ind) for ind in indices])


def get_subsampled_normalized_2chan(normalized_img, subs, plot=False):
    mask_files = glob("./masks/gen_masks/{}_0*".format(subs))
    mask_files = [np.fft.fftshift(np.load(m)) for m in mask_files]
    mask = mask_files[0]
    new_img = np.fft.ifft2(np.multiply(np.fft.fft2(normalized_img[:, :, 0]), mask))
    minv = np.min(np.real(new_img))
    maxv = np.max(np.real(new_img))
    normalized_new_img = np.zeros((320, 256, 2))
    normalized_new_img_real = -1 + (2 * (np.real(new_img) - minv) / (maxv - minv))
    normalized_new_img[:, :, 0] = normalized_new_img_real
    normalized_new_img[:, :, 1] = np.imag(new_img) 
    if plot:
        plt.imshow(np.sqrt(normalized_new_img[:, :, 0] ** 2 + normalized_new_img[:, :, 1] ** 2), cmap='gray')
        plt.show()
    #print(np.min(normalized_new_img))
    #print(np.max(normalized_new_img))
    return normalized_new_img

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

def recon_loss_L1(y_true, y_pred):
    mask_files = glob("./masks/gen_masks/2_0*")
    mask_files = [np.fft.fftshift(np.load(m)) for m in mask_files]
    mask = mask_files[0]
    y_pred = tf.squeeze(y_pred, axis=[0, 3])
    fft_img = tf.fft2d(tf.cast(y_pred, dtype=tf.complex64))
    masked = tf.multiply(fft_img, mask)
    ifft = tf.cast(tf.ifft2d(masked), dtype=tf.float32)
    squeeze_ifft = ifft #tf.squeeze(ifft, axis=0)
    squeeze_ytrue = tf.squeeze(y_true, axis=[0, 3])
    return K.mean(K.abs(squeeze_ifft - squeeze_ytrue), axis=-1) 

def recon_loss_L1_4(y_true, y_pred):
    mask_files = glob("./masks/gen_masks/4_0*")
    mask_files = [np.fft.fftshift(np.load(m)) for m in mask_files]
    mask = mask_files[0]
    y_pred = tf.squeeze(y_pred, axis=[0, 3])
    fft_img = tf.fft2d(tf.cast(y_pred, dtype=tf.complex64))
    masked = tf.multiply(fft_img, mask)
    ifft = tf.cast(tf.ifft2d(masked), dtype=tf.float32)
    squeeze_ifft = ifft #tf.squeeze(ifft, axis=0)
    squeeze_ytrue = tf.squeeze(y_true, axis=[0, 3])
    return K.mean(K.abs(squeeze_ifft - squeeze_ytrue), axis=-1) 

def recon_loss_L1_6(y_true, y_pred):
    mask_files = glob("./masks/gen_masks/6_0*")
    mask_files = [np.fft.fftshift(np.load(m)) for m in mask_files]
    mask = mask_files[0]
    y_pred = tf.squeeze(y_pred, axis=[0, 3])
    fft_img = tf.fft2d(tf.cast(y_pred, dtype=tf.complex64))
    masked = tf.multiply(fft_img, mask)
    ifft = tf.cast(tf.ifft2d(masked), dtype=tf.float32)
    squeeze_ifft = ifft #tf.squeeze(ifft, axis=0)
    squeeze_ytrue = tf.squeeze(y_true, axis=[0, 3])
    return K.mean(K.abs(squeeze_ifft - squeeze_ytrue), axis=-1) 

def recon_loss_L2(y_true, y_pred):
    mask_files = glob("./masks/gen_masks/2_0*")
    mask_files = [np.fft.fftshift(np.load(m)) for m in mask_files]
    mask = mask_files[0]
    y_pred = tf.squeeze(y_pred, axis=[0, 3])
    fft_img = tf.fft2d(tf.cast(y_pred, dtype=tf.complex64))
    masked = tf.multiply(fft_img, mask)
    ifft = tf.cast(tf.ifft2d(masked), dtype=tf.float32)
    squeeze_ifft = ifft #tf.squeeze(ifft, axis=0)
    squeeze_ytrue = tf.squeeze(y_true, axis=[0, 3])
    return K.mean(K.square(squeeze_ifft - squeeze_ytrue), axis=-1) # + 0.0000000001 * tf.norm(tf.reshape(squeeze_ifft, [-1]), ord=1) #+ 0.0000001 * tf.image.total_variation(tf.expand_dims(squeeze_ifft, -1))

def recon_loss_Linf(y_true, y_pred):
    mask_files = glob("./masks/gen_masks/2_0*")
    mask_files = [np.fft.fftshift(np.load(m)) for m in mask_files]
    mask = mask_files[0]
    y_pred = tf.squeeze(y_pred, axis=[0, 3])
    fft_img = tf.fft2d(tf.cast(y_pred, dtype=tf.complex64))
    masked = tf.multiply(fft_img, mask)
    ifft = tf.cast(tf.ifft2d(masked), dtype=tf.float32)
    squeeze_ifft = ifft #tf.squeeze(ifft, axis=0)
    squeeze_ytrue = tf.squeeze(y_true, axis=[0, 3])
    return K.max(K.abs(squeeze_ifft - squeeze_ytrue), axis=-1) # + 0.0000000001 * tf.norm(tf.reshape(squeeze_ifft, [-1]), ord=1) #+ 0.0000001 * tf.image.total_variation(tf.expand_dims(squeeze_ifft, -1))

def recon_loss_L2_2chan(y_true, y_pred):
    mask_files = glob("./masks/gen_masks/2_0*")
    mask_files = [np.fft.fftshift(np.load(m)) for m in mask_files]
    mask = mask_files[0]
    y_pred = tf.cast(tf.squeeze(y_pred, axis=0), dtype=tf.complex64)
    fft_img = tf.fft2d(y_pred[:, :, 0] + 1j*y_pred[:, :, 1])
    masked = tf.multiply(fft_img, mask)
    ifft = tf.ifft2d(masked)
    squeeze_ifft = tf.stack([tf.real(ifft), tf.imag(ifft)], axis=2)
    squeeze_ytrue = tf.squeeze(y_true, axis=0)
    return K.mean(K.square(squeeze_ifft - squeeze_ytrue), axis=-1) # + 0.0000000001 * tf.norm(tf.reshape(squeeze_ifft, [-1]), ord=1) #+ 0.0000001 * tf.image.total_variation(tf.expand_dims(squeeze_ifft, -1))

def recon_loss_L1_2chan(y_true, y_pred):
    mask_files = glob("./masks/gen_masks/4_0*")
    mask_files = [np.fft.fftshift(np.load(m)) for m in mask_files]
    mask = mask_files[0]
    y_pred = tf.cast(tf.squeeze(y_pred, axis=0), dtype=tf.complex64)
    fft_img = tf.fft2d(y_pred[:, :, 0] + 1j*y_pred[:, :, 1])
    masked = tf.multiply(fft_img, mask)
    ifft = tf.ifft2d(masked)
    squeeze_ifft = tf.stack([tf.real(ifft), tf.imag(ifft)], axis=2)
    squeeze_ytrue = tf.squeeze(y_true, axis=0)
    return K.mean(K.abs(squeeze_ifft - squeeze_ytrue), axis=-1) # + 0.0000000001 * tf.norm(tf.reshape(squeeze_ifft, [-1]), ord=1) #+ 0.0000001 * tf.image.total_variation(tf.expand_dims(squeeze_ifft, -1))




L1_loss = lambda x, y : (abs(x - y)).mean(axis=None)
L2_loss = lambda x, y : ((x - y)**2).mean(axis=None)

def train_network(network, noise, y, epochs, iterations, ground_truth, losses, jitter_schedule = [0], plot=False, batch_size=1):
    #base_image = np.expand_dims(noise, 0) 
    base_image = np.expand_dims(noise, 0) if np.shape(noise)[0] == 320 else noise
    y_image = np.expand_dims(y, 0) if np.shape(noise)[0] == 320 else y
    fit_params = {
        'x': base_image,
        'y': y_image, 
        'epochs': epochs,
        'batch_size': batch_size,
        'verbose': 0
    }
    if plot:
        plt.axis('off')
        plt.title('Iteration 0')
        plt.imshow(base_image[0, :, :, 0], cmap='gray')
        plt.show()
        plt.imshow(y_image[0, :, :, 0], cmap='gray')
        plt.show()
        
    results = np.empty(base_image.shape)
    all_losses = [[] for _ in range(len(losses))]
    
    if len(jitter_schedule) == 1:
        jitter = jitter_schedule[0]
        jitter_schedule = [jitter] * iterations

    noise_size = (batch_size, 320, 256, 1)
    for i in range(iterations):
        fit_params['x'] += jitter_schedule[i] * (np.random.random(size = noise_size) * 2 - 1) #(1, ) + np.shape(noise)
        network.fit(**fit_params)
        output = network.predict(base_image)
        results = np.append(results, output, axis=0)
        for j in range(len(losses)):
            all_losses[j].append(losses[j](output, ground_truth))
        if plot:
            plt.axis('off')
            plt.title('Iteration '+ str((i+1)*fit_params['epochs']))
            plt.imshow(output[0][:, :, 0], cmap='gray')
            plt.show()
    return results, all_losses, fit_params['x']

def train_network_2chan(network, noise, y, epochs, iterations, ground_truth, losses, jitter_schedule = [0], plot=False, batch_size=1):
    #base_image = np.expand_dims(noise, 0) 
    base_image = np.expand_dims(noise, 0) if np.shape(noise)[0] == 320 else noise
    y_image = np.expand_dims(y, 0) if np.shape(noise)[0] == 320 else y
    fit_params = {
        'x': base_image,
        'y': y_image, 
        'epochs': epochs,
        'batch_size': batch_size,
        'verbose': 0
    }
    if plot:
        plt.axis('off')
        plt.title('Iteration 0')
        plt.imshow(np.sqrt(base_image[0, :, :, 0]**2 + base_image[0, :, :, 1]**2), cmap='gray')
        plt.show()
        plt.imshow(np.sqrt(y_image[0, :, :, 0]**2 + y_image[0, :, :, 1]**2), cmap='gray')
        plt.show()
        
    results = np.empty(base_image.shape)
    all_losses = [[] for _ in range(len(losses))]
    
    if len(jitter_schedule) == 1:
        jitter = jitter_schedule[0]
        jitter_schedule = [jitter] * iterations

    noise_size = (batch_size, 320, 256, 2)
    for i in range(iterations):
        fit_params['x'] += jitter_schedule[i] * (np.random.random(size = noise_size) * 2 - 1) #(1, ) + np.shape(noise)
        network.fit(**fit_params)
        output = network.predict(base_image)
        results = np.append(results, output, axis=0)
        for j in range(len(losses)):
            all_losses[j].append(losses[j](output, ground_truth))
        if plot:
            plt.axis('off')
            plt.title('Iteration '+ str((i+1)*fit_params['epochs']))
            plt.imshow(np.sqrt(output[0][:, :, 0] ** 2 + output[0][:, :, 1] ** 2), cmap='gray')
            plt.show()
    return results, all_losses, fit_params['x']

# 2chan - test image progression after full train with batch

trials = 3
all_results = [[] for i in range(4)] # same image, diff image, 2 diff, 4 diff
for i in range(trials):
    noisy = np.random.random(size=(320, 256, 2)) * 2 - 1
    test_noise = np.random.random(size=(320, 256, 2)) * 2 - 1
    epochs = 70
    epochs1 = 70
    iters1 = 25
    iters = 50

    batch_size = 1
    print('start 1')
    noise = np.array([noisy for _ in range(batch_size)])
    jit_sched = [0.1 if k < iters // 2 else 0.01 for k in range(iters)]
    
    normalized_img = get_batch_normalized_2chan([100])
    normalized_new_img = get_subsampled_normalized_2chan(get_image_normalized_2chan(100), 4)
    autoencoder = define_network_4_2chan(verbose=False)
    autoencoder.compile(Adam(1e-3), loss = 'mae')
    results, losses, new_noise = train_network_2chan(autoencoder, noise, normalized_img, epochs1, iters1, normalized_img, [L1_loss, L2_loss], jitter_schedule=jit_sched, plot=False, batch_size=batch_size)

    autoencoder.compile(Adam(1e-3), loss = recon_loss_L1_2chan)
    jit_sched = [0.1 if k < iters // 2 else 0.01 for k in range(iters)]
    normalized_img = get_image_normalized_2chan(100)
    results, losses, _ = train_network_2chan(autoencoder, test_noise, normalized_new_img, epochs, iters, normalized_img, [L1_loss, L2_loss], jitter_schedule=jit_sched, plot=False)
    autoencoder.compile(Adam(1e-4), loss = recon_loss_L1_2chan)
    results, losses, _ = train_network_2chan(autoencoder, test_noise, normalized_new_img, epochs, iters, normalized_img, [L1_loss, L2_loss], jitter_schedule=jit_sched, plot=False)
    all_results[0].append((results, losses))
    
    del autoencoder
    gc.collect()

    batch_size = 1
    print('start 2')
    noise = np.array([noisy for _ in range(batch_size)])
    jit_sched = [0.1 if k < iters // 2 else 0.01 for k in range(iters)]
    
    normalized_img = get_batch_normalized_2chan([170])
    normalized_new_img = get_subsampled_normalized_2chan(get_image_normalized_2chan(100), 4)
    autoencoder = define_network_4_2chan(verbose=False)
    autoencoder.compile(Adam(1e-3), loss = 'mae')
    results, losses, new_noise = train_network_2chan(autoencoder, noise, normalized_img, epochs1, iters1, normalized_img, [L1_loss, L2_loss], jitter_schedule=jit_sched, plot=False, batch_size=batch_size)

    autoencoder.compile(Adam(1e-3), loss = recon_loss_L1_2chan)
    jit_sched = [0.1 if k < iters // 2 else 0.01 for k in range(iters)]
    normalized_img = get_image_normalized_2chan(100)
    results, losses, _ = train_network_2chan(autoencoder, test_noise, normalized_new_img, epochs, iters, normalized_img, [L1_loss, L2_loss], jitter_schedule=jit_sched, plot=False)
    autoencoder.compile(Adam(1e-4), loss = recon_loss_L1_2chan)
    results, losses, _ = train_network_2chan(autoencoder, test_noise, normalized_new_img, epochs, iters, normalized_img, [L1_loss, L2_loss], jitter_schedule=jit_sched, plot=False)
    all_results[1].append((results, losses))
    
    del autoencoder
    gc.collect()

    batch_size = 2
    print('start 3')
    #noise = np.array([noisy for _ in range(batch_size)])
    noise = np.array([np.random.random(size=(320, 256, 2)) * 2 - 1 for _ in range(batch_size)])
    jit_sched = [0.1 if k < iters // 2 else 0.01 for k in range(iters)]
    
    normalized_img = get_batch_normalized_2chan([170, 240])
    normalized_new_img = get_subsampled_normalized_2chan(get_image_normalized_2chan(100), 4)
    autoencoder = define_network_4_2chan(verbose=False)
    autoencoder.compile(Adam(1e-3), loss = 'mae')
    results, losses, new_noise = train_network(autoencoder, noise, normalized_img, epochs1, iters1, normalized_img, [L1_loss, L2_loss], jitter_schedule=jit_sched, plot=False, batch_size=batch_size)

    autoencoder.compile(Adam(1e-3), loss = recon_loss_L1_2chan)
    jit_sched = [0 if iters % 10 != 0 else 0.1 if k < iters // 2 else 0.01 for k in range(iters)]  #[0.1 if k < iters // 2 else 0.01 for k in range(iters)]
    normalized_img = get_image_normalized_2chan(100)
    results, losses, _ = train_network_2chan(autoencoder, test_noise, normalized_new_img, epochs, iters, normalized_img, [L1_loss, L2_loss], jitter_schedule=jit_sched, plot=False)
    autoencoder.compile(Adam(1e-4), loss = recon_loss_L1_2chan)
    results, losses, _ = train_network_2chan(autoencoder, test_noise, normalized_new_img, epochs, iters, normalized_img, [L1_loss, L2_loss], jitter_schedule=jit_sched, plot=False)
    all_results[2].append((results, losses))
    
    del autoencoder
    gc.collect()

    batch_size = 4
    print('start 4')
    #noise = np.array([noisy for _ in range(batch_size)])
    noise = np.array([np.random.random(size=(320, 256, 2)) * 2 - 1 for _ in range(batch_size)])
    jit_sched = [0.1 if k < iters // 2 else 0.01 for k in range(iters)]
    
    normalized_img = get_batch_normalized_2chan([70, 170, 210, 240])
    normalized_new_img = get_subsampled_normalized_2chan(get_image_normalized_2chan(100), 4)
    autoencoder = define_network_4_2chan(verbose=False)
    autoencoder.compile(Adam(1e-3), loss = 'mae')
    results, losses, new_noise = train_network_2chan(autoencoder, noise, normalized_img, epochs1, iters1, normalized_img, [L1_loss, L2_loss], jitter_schedule=jit_sched, plot=False, batch_size=batch_size)

    autoencoder.compile(Adam(1e-3), loss = recon_loss_L1_2chan)
    jit_sched = [0 if iters % 10 != 0 else 0.1 if k < iters // 2 else 0.01 for k in range(iters)]  #[0.1 if k < iters // 2 else 0.01 for k in range(iters)]
    normalized_img = get_image_normalized_2chan(100)
    results, losses, _ = train_network_2chan(autoencoder, test_noise, normalized_new_img, epochs, iters, normalized_img, [L1_loss, L2_loss], jitter_schedule=jit_sched, plot=False) # was new_noise[0]
    autoencoder.compile(Adam(1e-4), loss = recon_loss_L1_2chan)
    results, losses, _ = train_network_2chan(autoencoder, test_noise, normalized_new_img, epochs, iters, normalized_img, [L1_loss, L2_loss], jitter_schedule=jit_sched, plot=False)
    all_results[3].append((results, losses))
    
    del autoencoder
    gc.collect()
    

np.save('diff_batch_sizes_2chan_L1', all_results)