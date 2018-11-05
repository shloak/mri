
# coding: utf-8

# In[ ]:


import os
import scipy.misc
import numpy as np
import cv2
import ra

from forward_model import DCGAN
from utils import pp, visualize, to_json, show_all_variables

import tensorflow as tf

from glob import glob
from ops import *
from utils import *

import matplotlib.pyplot as plt

def show_images(images, cols = 1):
    n_images = len(images)
    fig = plt.figure()
    for n, image in enumerate(images):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        plt.imshow(image, cmap='gray')
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()
    
height = 320  
width = 256   


# In[ ]:


#data = glob("./celebA/*.jpg")[3005:3009] #face
data = glob("./data/valid_img_slices/*.ra")         #mri
#np.random.shuffle(data)

sample_files = data[210:214]
#sample = [get_image_old1(sample_file, True) for sample_file in sample_files] #face
#sample = [cv2.resize(s, dsize=(256, 320)) for s in sample]                   #face
sample = [get_image_old2(sample_file) for sample_file in sample_files]        #mri


# In[ ]:


plt.imshow(np.sqrt(sample[0][:, :, 0]**2 + sample[0][:, :, 1]**2), cmap ='gray')
plt.show()
plt.imshow(sample[0][:, :, 1], cmap ='gray')
plt.show()


# In[ ]:


#v = np.reshape(sample, (4, height*width))

v = np.reshape(sample, (4, height*width*2))[3:4]

# number of features per image (pixels including all channels)
print('shape of v: {}'.format(v.shape))


n = v.shape[1]


# In[ ]:


# [4x, 3x, 6x, 8x, 5x, 7x, 2x] AZURE VM
# [2x, 3x, 4x, 5x, 6x, 7x, 8x] DELL COMP

masks = glob("./masks/masks/*.ra")

img = ra.read_ra(masks[6]).T
mag = abs(img)
mag = (mag > 0.5)*1.

the_mask = np.fft.fftshift(mag)
print(np.shape(the_mask))
m = sum(sum(the_mask))
print(m)


# In[ ]:


comb_sample = sample[0][:, :, 0] + 1j*sample[0][:, :, 1]

kspace = np.fft.fft2(comb_sample) / np.sqrt(height*width)
kspace = np.multiply(kspace, the_mask) # if subsample

inv = np.fft.ifft2(kspace)
print(np.shape(inv))
plt.imshow(np.sqrt(np.real(inv)**2 + np.imag(inv)**2), cmap='gray')
plt.show()
#plt.imshow(np.concatenate([np.real(inv), np.imag(inv)]), cmap='gray')
#plt.show()

final_image = np.zeros((320, 256, 2))
final_image[:, :, 0] = np.real(inv)
final_image[:, :, 1] = np.imag(inv)

'''plt.imshow(final_image[:, :, 0], cmap='gray')
plt.show()
maxv, minv = np.amax(final_image[:, :, 0]), np.amin(final_image[:, :, 0])
final_image[:, :, 0] = (final_image[:, :, 0] - minv)/ (maxv - minv)
maxv, minv = np.amax(final_image[:, :, 1]), np.amin(final_image[:, :, 1])
final_image[:, :, 1] = (final_image[:, :, 1] - minv)/ (maxv - minv)
plt.imshow(final_image[:, :, 0], cmap='gray')
plt.show()
print(np.amax(final_image[:, :, 0]), np.amin(final_image[:, :, 0]))'''

kspace = np.reshape(kspace, (-1))
m_ = np.shape(kspace)[0]
print(m_)


# In[ ]:


num_iter = 4000
with tf.Session() as sess:
    mask= tf.cast(the_mask, dtype=tf.complex64)
    y_placeholder2 = tf.placeholder(tf.complex64,[m_])
    X = tf.Variable(tf.random_uniform([320, 256, 2]))
    X2 = tf.squeeze(tf.cast(X, dtype=tf.complex64))
    X_fft = X2[:, :, 0] + 1j*X2[:, :, 1]
    X_fft = tf.fft2d(X_fft) 
    X_fft = tf.multiply(X_fft, mask) / np.sqrt(height*width) # if subsample
    #X_fft = tf.ifft2d(X_fft)
    #X_fft = tf.stack([tf.real(X_fft), tf.imag(X_fft)], axis=2)
    #X_fft = tf.stack([(X_fft[:, :, 0]- tf.reduce_min(X_fft[:, :, 0])) / (tf.reduce_max(X_fft[:, :, 0]) - tf.reduce_min(X_fft[:, :, 0])), \
    #(X_fft[:, :, 1]- tf.reduce_min(X_fft[:, :, 1])) / (tf.reduce_max(X_fft[:, :, 1]) - tf.reduce_min(X_fft[:, :, 1]))])
    X_fft = tf.cast(tf.reshape(X_fft, [-1]), dtype=tf.complex64) 
    our_loss = (tf.reduce_sum(tf.abs(X_fft - y_placeholder2)**2))
    lp_loss = tf.norm(X, ord=2)
    tv_loss = tf.image.total_variation(X)
    z_optim = tf.train.AdamOptimizer(learning_rate=0.01).minimize(our_loss + 10 * lp_loss + .005 * tv_loss)
    init=tf.global_variables_initializer()
    sess.run(init)
    
    for i in range(num_iter):
        a, loss = sess.run([z_optim, our_loss],feed_dict={y_placeholder2: kspace})
        if np.mod(i, 100) == 1: 
            print('loss:', loss, ' iter:', i)
            x_curr = X.eval()
            plt.imshow(x_curr[:, :, 0], cmap='gray')
            plt.show()
            x_comb = x_curr[:, :, 0] + 1j*x_curr[:, :, 1]
            x_masked = np.fft.ifft2(np.multiply(np.fft.fft2(x_comb) / np.sqrt(height * width), the_mask))
            #plt.imshow(np.real(x_masked).reshape((320, 256)), cmap='gray')
            print(np.amax(x_curr[:, :, 0]), np.amin(x_curr[:, :, 0]))
            #plt.show()
            #plt.imshow(np.real(inv), cmap='gray')
            #print(np.amax(np.real(inv)), np.amin(np.real(inv)))
            #plt.show()
    print('final masked')
    plt.imshow(np.real(x_masked).reshape((320, 256)), cmap='gray')
    plt.show()

