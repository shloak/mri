"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
from skimage.util.shape import view_as_windows
from sklearn.feature_extraction import image
from skimage import color
from skimage import io
from glob import glob
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime
from six.moves import xrange

import ra
import os
import struct
import warnings

import tensorflow as tf
import tensorflow.contrib.slim as slim

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])


def show_all_variables():
  model_vars = tf.trainable_variables()
  slim.model_analyzer.analyze_vars(model_vars, print_info=True)

# gets image patches for one images, return size is [16, 64, 80], puts between [-1, 1]    
def get_image(image_path, input_height=256, input_width=320,
              resize_height=64, resize_width=80,
              crop=True, grayscale=False):
  return patch_tf(get_image_old(image_path), resize_height, resize_width)
 
# gets one image from path, does not patch, puts between [-1, 1]  CURRENT METHOD FOR MRI  
def get_image_old2(image_path):
  '''image = imread(image_path)
  a = (np.array(image) - 0.5) / 0.5 # *2 - 1
  return np.reshape(a, (320, 256, 1))'''
  img = (ra.read_ra(os.path.join(image_path))).T
  test = np.zeros((320, 256, 2))
  test[:, :, 0] = np.real(img) #(np.real(img) - 0.5) / 0.5
  test[:, :, 1] = np.imag(img) #(np.imag(img) - 0.5) / 0.5 #
  return test

# converts single channel to dual channel
def convert(img):
  '''test = np.zeros((320, 256, 2))
  test[:, :, 0] = tf.real(img)
  test[:, :, 1] = tf.imag(img)
  return tf.convert_to_tensor(test, dtype=tf.float32)'''
  return tf.stack([tf.real(img), tf.imag(img)], axis=2)

def get_image_old1(image_path, grayscale): # only for celebA images
  im = color.rgb2gray(io.imread(image_path))
  return (im - 0.5) / 0.5

# gets one image from path, resizes by size factor, puts between [-1, 1]    
def get_image_old(image_path, size, input_height=256, input_width=320,
              resize_height=256, resize_width=320,
              crop=False, grayscale=False):
  image = imread(image_path, grayscale)
  cropped_image = center_crop(
      image, input_height, input_width, 
      resize_height, resize_width)
  return scipy.misc.imresize(np.array(cropped_image)/127.5 - 1, 0.25) # *2 - 1

# patches into 64x80 patches
def patch_tf(image_, height, width):
  #with tf.Session() as sess:
  img = (tf.extract_image_patches(images=tf.expand_dims(tf.expand_dims(image_, 0), 3), ksizes=[1, height, width, 1], 
                                  strides=[1, height, width, 1], rates=[1, 1, 1, 1], padding='VALID').eval())
  img = np.reshape(img, ((256*320)//(height*width), height, width))
  return img    

def get_mask():
  masks = glob("./masks/masks/*.ra")
  img = ra.read_ra(masks[6]).T
  mag = abs(img)
  mag = (mag > 0.5)*1.
  the_mask = np.fft.fftshift(mag)
  return the_mask
    
def patch_new(path, height=64, width=80):
  #with tf.Session() as sess:
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

def imread(path, grayscale = False):
  return abs(ra.read_ra(os.path.join(path))).T

def imread_new(path, grayscale = False):
  return np.load(path)

def imread_old(path, grayscale = True):
  if (grayscale):
    return scipy.misc.imread(path, flatten = True).astype(np.float)
  else:
    return scipy.misc.imread(path).astype(np.float)

def merge_images(images, size):
  return inverse_transform(images)

def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  if (images.shape[3] in (3,4)):
    c = images.shape[3]
    img = np.zeros((h * size[0], w * size[1], c))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w, :] = image
    return img
  elif images.shape[3]==1:
    img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
    return img
  else:
    raise ValueError('in merge(images,size) images parameter '
                     'must have dimensions: HxW or HxWx3 or HxWx4')

def imsave(images, size, path):
  image = np.clip(np.squeeze(merge(images, size)), 0, 1)
  image.ravel()[0] = 0
  image.ravel()[-1] = 1
  return scipy.misc.imsave(path, image)


def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
  if crop_w is None:
    crop_w = crop_h
  h, w = x.shape[:2]
  j = int(round((h - crop_h)/2.))
  i = int(round((w - crop_w)/2.))
  return scipy.misc.imresize(
      x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def transform(image, input_height, input_width, 
              resize_height=64, resize_width=64, crop=True):
  if crop:
    cropped_image = center_crop(
      image, input_height, input_width, 
      resize_height, resize_width)
  else:
    cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
  return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
  #return (images+1.)/2.
  c_image = (images * 0.5) + 0.5
  return np.expand_dims(c_image[:, :, :, 0], axis=-1)
  '''c_image = tf.stack([tf.sqrt(tf.square(c_image[0, :, :, 0]) + tf.square(c_image[0, :, :, 1])), tf.sqrt(tf.square(c_image[1, :, :, 0]) + tf.square(c_image[1, :, :, 1])), tf.sqrt(tf.square(c_image[2, :, :, 0]) + tf.square(c_image[2, :, :, 1])), tf.sqrt(tf.square(c_image[3, :, :, 0]) + tf.square(c_image[3, :, :, 1]))])'''
  return c_image

def to_json(output_path, *layers):
  with open(output_path, "w") as layer_f:
    lines = ""
    for w, b, bn in layers:
      layer_idx = w.name.split('/')[0].split('h')[1]

      B = b.eval()

      if "lin/" in w.name:
        W = w.eval()
        depth = W.shape[1]
      else:
        W = np.rollaxis(w.eval(), 2, 0)
        depth = W.shape[0]

      biases = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(B)]}
      if bn != None:
        gamma = bn.gamma.eval()
        beta = bn.beta.eval()

        gamma = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(gamma)]}
        beta = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(beta)]}
      else:
        gamma = {"sy": 1, "sx": 1, "depth": 0, "w": []}
        beta = {"sy": 1, "sx": 1, "depth": 0, "w": []}

      if "lin/" in w.name:
        fs = []
        for w in W.T:
          fs.append({"sy": 1, "sx": 1, "depth": W.shape[0], "w": ['%.2f' % elem for elem in list(w)]})

        lines += """
          var layer_%s = {
            "layer_type": "fc", 
            "sy": 1, "sx": 1, 
            "out_sx": 1, "out_sy": 1,
            "stride": 1, "pad": 0,
            "out_depth": %s, "in_depth": %s,
            "biases": %s,
            "gamma": %s,
            "beta": %s,
            "filters": %s
          };""" % (layer_idx.split('_')[0], W.shape[1], W.shape[0], biases, gamma, beta, fs)
      else:
        fs = []
        for w_ in W:
          fs.append({"sy": 5, "sx": 5, "depth": W.shape[3], "w": ['%.2f' % elem for elem in list(w_.flatten())]})

        lines += """
          var layer_%s = {
            "layer_type": "deconv", 
            "sy": 5, "sx": 5,
            "out_sx": %s, "out_sy": %s,
            "stride": 2, "pad": 1,
            "out_depth": %s, "in_depth": %s,
            "biases": %s,
            "gamma": %s,
            "beta": %s,
            "filters": %s
          };""" % (layer_idx, 2**(int(layer_idx)+2), 2**(int(layer_idx)+2),
               W.shape[0], W.shape[3], biases, gamma, beta, fs)
    layer_f.write(" ".join(lines.replace("'","").split()))

def make_gif(images, fname, duration=2, true_image=False):
  import moviepy.editor as mpy

  def make_frame(t):
    try:
      x = images[int(len(images)/duration*t)]
    except:
      x = images[-1]

    if true_image:
      return x.astype(np.uint8)
    else:
      return ((x+1)/2*255).astype(np.uint8)

  clip = mpy.VideoClip(make_frame, duration=duration)
  clip.write_gif(fname, fps = len(images) / duration)

def visualize(sess, dcgan, config, option):
  image_frame_dim = int(math.ceil(config.batch_size**.5))
  if option == 0:
    z_sample = np.random.uniform(-0.5, 0.5, size=(config.batch_size, dcgan.z_dim))
    samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
    save_images(samples, [image_frame_dim, image_frame_dim], './sample_final/test_%s.png' % strftime("%Y%m%d%H%M%S", gmtime()))
  elif option == 1:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in xrange(100):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      if config.dataset == "mnist":
        y = np.random.choice(10, config.batch_size)
        y_one_hot = np.zeros((config.batch_size, 10))
        y_one_hot[np.arange(config.batch_size), y] = 1

        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_one_hot})
      else:
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})

      save_images(samples, [image_frame_dim, image_frame_dim], './samples/test_arange_%s.png' % (idx))
  elif option == 2:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in [random.randint(0, 99) for _ in xrange(100)]:
      print(" [*] %d" % idx)
      z = np.random.uniform(-0.2, 0.2, size=(dcgan.z_dim))
      z_sample = np.tile(z, (config.batch_size, 1))
      #z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      if config.dataset == "mnist":
        y = np.random.choice(10, config.batch_size)
        y_one_hot = np.zeros((config.batch_size, 10))
        y_one_hot[np.arange(config.batch_size), y] = 1

        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_one_hot})
      else:
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})

      try:
        make_gif(samples, './samples/test_gif_%s.gif' % (idx))
      except:
        save_images(samples, [image_frame_dim, image_frame_dim], './samples/test_%s.png' % strftime("%Y%m%d%H%M%S", gmtime()))
  elif option == 3:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in xrange(100):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
      make_gif(samples, './samples/test_gif_%s.gif' % (idx))
  elif option == 4:
    image_set = []
    values = np.arange(0, 1, 1./config.batch_size)

    for idx in xrange(100):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample): z[idx] = values[kdx]

      image_set.append(sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample}))
      make_gif(image_set[-1], './samples/test_gif_%s.gif' % (idx))

    new_image_set = [merge(np.array([images[idx] for images in image_set]), [10, 10]) \
        for idx in range(64) + range(63, -1, -1)]
    make_gif(new_image_set, './samples/test_gif_merged.gif', duration=8)


def image_manifold_size(num_images):
  manifold_h = int(np.floor(np.sqrt(num_images)))
  manifold_w = int(np.ceil(np.sqrt(num_images)))
  assert manifold_h * manifold_w == num_images
  return manifold_h, manifold_w
