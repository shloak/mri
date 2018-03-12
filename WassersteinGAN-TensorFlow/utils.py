from __future__ import division
import os
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime
from glob import glob
import ra
import os
import struct
import warnings

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

# gets image patches for one images, return size is [16, 64, 80], puts between [-1, 1]    
def get_image(image_path,
              resize_height=64, resize_width=80):
  return patch_tf(get_image_old(image_path), resize_height, resize_width)
 
# gets one image from path, does not patch, puts between [-1, 1]    
def get_image_old1(image_path):
  image = imread(image_path)
  return np.array(image)/127.5 - 1 # *2 - 1

# gets one image from path, resizes by size factor, puts between [-1, 1]    
def get_image_old(image_path, size):
  image = imread(image_path)
  return scipy.misc.imresize(np.array(image/127.5 - 1, size)) # *2 - 1

# patches into 64x80 patches
def patch_tf(image_, height, width):
  img = (tf.extract_image_patches(images=tf.expand_dims(tf.expand_dims(image_, 0), 3), ksizes=[1, height, width, 1], 
                                  strides=[1, height, width, 1], rates=[1, 1, 1, 1], padding='VALID').eval())
  img = np.reshape(img, ((256*320)//(height*width), height, width))
  return img    
    
def patch_new(path, height=64, width=80):
   image_ = get_image_old(path)
   img = (tf.space_to_batch_nd(input=tf.expand_dims(tf.expand_dims(image_, 0), 3), block_shape=[64, 80], 
                                   paddings=[[0, 0], [0, 0]]).eval())
   img = np.reshape(img, (16, 64, 80))
   return img 
# pieces back the patches into one image of size (256, 320). requires as input (16, 64, 80) or (4, 4, 64, 80)
def patch_together(img, height, width):    
    img = np.reshape(img, (256//height, 320//width, height, width)) # rows, cols, height, width
    x, y = np.shape(img)[0], np.shape(img)[1]
    horiz = []
    for i in range(x):
        horiz.append(np.hstack((img[i, :])))
    return np.vstack(horiz[:]) 

def addZeros(epoch, idx):
    num = epoch*10 + idx
    while len(str(num)) < 10:
        num = str('0' + str(num))
    return num

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path):
  return abs(ra.read_ra(os.path.join(path)))

def imread_new(path):
  return np.load(path)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def center_crop(x, crop_h, crop_w=None, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
                               [resize_w, resize_w])

def transform(image, npx=64, is_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.

def get_batch_images(batch_index, data, config):
    batch_files = data[batch_index*config.batch_size:(batch_index+1)*config.batch_size]
    batch = [get_image_old1(batch_file) for batch_file in batch_files]
    batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
    return batch_images

def get_data_arr():
    return glob("./data/train_img_slices/*.ra")[0:8]