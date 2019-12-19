import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import gc

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, MaxPooling2D, BatchNormalization, LeakyReLU
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from deep_prior_networks import *
import tensorflow.keras.backend as K
from glob import glob
import sigpy as sp
import sigpy.mri as mr
import ipdb
import cv2
import time
import ra
import os
current_milli_time = lambda: int(round(time.time() * 1000))

def get_image_old2(image_path):
  '''image = imread(image_path)
  a = (np.array(image) - 0.5) / 0.5 # *2 - 1
  return np.reshape(a, (320, 256, 1))'''
  img = (ra.read_ra(os.path.join(image_path))).T
  test = np.zeros((320, 256, 2))
  test[:, :, 0] = np.real(img) #(np.real(img) - 0.5) / 0.5
  test[:, :, 1] = np.imag(img) #(np.imag(img) - 0.5) / 0.5 #
  return test



L1_loss = lambda x, y : (abs(x - y)).mean(axis=None)
L2_loss = lambda x, y : ((x - y)**2).mean(axis=None)


def get_image_normalized_2chan(index, plot=False):
    img = get_image_old2('./data/test_img_slices/19_{}.ra'.format(index)) # same - for linux
    minv = np.min(img[:, :, 0])
    maxv = np.max(img[:, :, 0])
    normalized_img = np.zeros((320, 256, 2))
    
    normalized_img_real = -1 + (2 * (np.array(img[:, :, 0] - minv) / (maxv - minv)))
    normalized_img[:, :, 0] = normalized_img_real
    
    
    minv = np.min(img[:, :, 1])
    maxv = np.max(img[:, :, 1])
    
    normalized_img_imag = -1 + (2 * (np.array(img[:, :, 1] - minv) / (maxv - minv)))
    normalized_img[:, :, 1] = normalized_img_imag
    
    
    #normalized_img[:, :, 1] = img[:, :, 1]
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

def get_subsampled_normalized_2chan2(normalized_img, mask, plot=False):
    x = mask.shape
    fft_img = np.fft.fft2(normalized_img[:, :, 0] + 1j * normalized_img[:, :, 1])
    masked_img = np.multiply(fft_img , mask)
    new_img = np.fft.ifft2(masked_img)
    minv = np.min(np.real(new_img))
    maxv = np.max(np.real(new_img))
    normalized_new_img = np.zeros(normalized_img.shape)
    normalized_new_img_real = -1 + (2 * (np.real(new_img) - minv) / (maxv - minv))
    normalized_new_img[:, :, 0] = np.real(new_img)
    normalized_new_img[:, :, 1] = np.imag(new_img)
    if plot:
        
        plt.imshow(-np.sqrt(normalized_new_img[:, :, 0] ** 2 + normalized_new_img[:, :, 1] ** 2), cmap='gray')
        plt.show()
    return normalized_new_img

#l1 wavelet


def train_network_2chan(network, noise, y, epochs, iterations, ground_truth, losses, jitter_schedule = [0], plot=False, batch_size=1 , exp_name="test"):
    #base_image = np.expand_dims(noise, 0) 
    base_image = np.expand_dims(noise, 0) if np.shape(noise)[0] > 1 else noise
    y_image = np.expand_dims(y, 0) if np.shape(noise)[0] > 1 else y
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

    noise_size = y.shape
    for i in range(iterations):
        print("Iteraton " , i)
        fit_params['x'] += jitter_schedule[i] * (np.random.random(size = noise_size) * 2 - 1) #(1, ) + np.shape(noise)
        network.fit(**fit_params)
        output = network.predict(base_image)
        results = np.append(results, output, axis=0)
        for j in range(len(losses)):
            all_losses[j].append(losses[j](output, ground_truth))

        print(all_losses[-1][-1])
        if plot:
            plt.axis('off')
            plt.title('Iteration '+ str((i+1)*fit_params['epochs']))
            plt.imshow(-np.sqrt(output[0][:, :, 0] ** 2 + output[0][:, :, 1] ** 2), cmap='gray')
            plt.show()
            get_subsampled_normalized_2chan2( output[0] , mask , True)

   
    fig = plt.figure()
    plt.title('Iteration '+ str((i+1)*fit_params['epochs']))
    plt.imshow(-np.sqrt(output[0][:, :, 0] ** 2 + output[0][:, :, 1] ** 2), cmap='gray')
    fig.savefig("./run_data/" + exp_name + str(current_milli_time()) +  ".png")
    np.save("./run_data/" + exp_name  + str(current_milli_time()), all_losses)
    plt.show()
                    
    return results, all_losses, fit_params['x']

mask_files = glob("./masks/gen_masks/{}_1*".format(4))
mask_files = [np.fft.fftshift(np.load(m)) for m in mask_files]
mask = mask_files[0]


# mask = np.load("./masks/ffast/ffast_mask_4x.npy")

# result = np.zeros([320 , 256])
# result[:mask.shape[0],:256] = mask[: , :256]
# mask = result

# for i in range(len(mask[1])):
#     if(mask[1][i] >0):
#         mask.T[i] = np.ones(len(mask))
# bar = 10
# mask[len(mask)//2-bar : len(mask)//2+bar] = np.ones([2*bar , mask.shape[1]])
# mask = mask.T
# mask[len(mask)//2-bar : len(mask)//2+bar] = np.ones([2*bar , mask.shape[1]])
# mask = mask.T
# mask = np.fft.fftshift(mask)

#mask = np.ones([320 , 256])

#mask = np.load("./masks/ffast/ffast_mask_4x.npy")

# mask = mask[:304 , :304]

# bar = 10
# mask[len(mask)//2-bar : len(mask)//2+bar] = np.ones([2*bar , mask.shape[1]])
# mask = mask.T
# mask[len(mask)//2-bar : len(mask)//2+bar] = np.ones([2*bar , mask.shape[1]])
# mask = mask.T
# mask = np.fft.fftshift(mask)


mask = mask.astype(complex)

def recon_loss_L1_2chan_fixed3(y_true, y_pred):
    y_pred = tf.cast(tf.squeeze(y_pred, axis=0), dtype=tf.complex64)
    fft_img = tf.fft2d(y_pred[:, :, 0] + 1j*y_pred[:, :, 1])
    masked = tf.multiply(fft_img, mask)
    ifft = tf.ifft2d(masked)    
    squeeze_ytrue =tf.cast( tf.squeeze(y_true, axis=0) , dtype=tf.complex64)

    #New conjugate loss function
    imag_ytrue = squeeze_ytrue[: ,: , 0] + 1j * squeeze_ytrue[: ,: , 1]
    subtract = tf.cast(ifft - imag_ytrue , tf.complex64)
    conj = tf.cast(tf.real(subtract), dtype=tf.complex64) -  1j * tf.cast(tf.imag(subtract), dtype=tf.complex64)
    loss = tf.sqrt(tf.real(tf.multiply(subtract , conj)))
    
    loss = tf.reduce_sum(loss)
    
    print("ifft" , ifft.shape)
    print("imag_ytrue" , imag_ytrue.shape)
    print("squeeze_ytrue" , squeeze_ytrue.shape)
    print("loss" , loss.shape)
    
    return loss



trials = 1
all_results = [] # same image, diff image, 2 diff, 4 diff
for i in range(trials):
    
    epochs = 70
    iters = 100
 
    batch_size = 1
    print('start 1')
   
    
    
    #----> NORMAL
    normalized_img = get_batch_normalized_2chan([100])
    normalized_img_sub = get_subsampled_normalized_2chan2(normalized_img[0], mask)
    normalized_img_sub = np.array([normalized_img_sub])
    #------
    
    
#     #----> FFAST
#     normalized_img = get_batch_normalized_2chan([100])[0]

#     result = np.zeros([304,304,2]) + np.mean(normalized_img[0, : , :] , axis=0)


#     result[:mask.shape[0],:normalized_img.shape[1], :] = normalized_img[:mask.shape[0], : , :]
#     normalized_img = result
#     normalized_img_sub = get_subsampled_normalized_2chan2(normalized_img, mask)
#     normalized_img = np.array([normalized_img])
#     normalized_img_sub = np.array([normalized_img_sub])
#     print(normalized_img_sub.shape)
#     print(normalized_img.shape)
    
#     # ------
    
    noisy = np.random.random(size=normalized_img.shape[1:]) * 2 - 1
    test_noise = np.random.random(size=normalized_img.shape[1:]) * 2 - 1
    
    noise = np.array([noisy for _ in range(batch_size)])
    jit_sched = [0.1 if k < iters // 2 else 0.01 for k in range(iters)]
    
    
    autoencoder = define_network_4_2chan(verbose=False)
    
    autoencoder.compile(Adam(5e-4), loss = recon_loss_L1_2chan_fixed3)
    results, losses, new_noise = train_network_2chan(autoencoder, noise, normalized_img_sub, epochs, iters, normalized_img, [L1_loss, L2_loss], jitter_schedule=jit_sched, plot=False, batch_size=batch_size , exp_name = "4xMask_noise")
    all_results.append(losses)

    del autoencoder
    gc.collect()

    autoencoder = define_network_4_2chan(verbose=False)
    
    autoencoder.compile(Adam(5e-4), loss = recon_loss_L1_2chan_fixed3)
    results, losses, new_noise = train_network_2chan(autoencoder, normalized_img_sub, normalized_img_sub, epochs, iters, normalized_img, [L1_loss, L2_loss], jitter_schedule=jit_sched, plot=False, batch_size=batch_size , exp_name = "4xMask_self")
    all_results.append(losses)

    del autoencoder
    gc.collect()


#     print("start 2")
#     noise = np.array([noisy for _ in range(batch_size)])
#     jit_sched = [0.1 if k < iters // 2 else 0.01 for k in range(iters)]
    
#     normalized_img = get_batch_normalized_2chan([100])
#     normalized_new_img = get_subsampled_normalized_2chan(get_image_normalized_2chan(100), 4)
#     autoencoder = define_network_4_2chan(verbose=False)
#     autoencoder.compile(Adam(1e-3), loss = recon_loss_L2_2chan_fixed3 )
#     results, losses, new_noise = train_network_2chan(autoencoder, noise, normalized_img, epochs, iters, normalized_img, [L1_loss, L2_loss], jitter_schedule=jit_sched, plot=False, batch_size=batch_size)
#     all_results.append(losses)
    
#     del autoencoder
#     gc.collect()