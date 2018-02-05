from skimage.util.shape import view_as_windows
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
import ra
import struct
import warnings


def get_image(image_path, input_height=64, input_width=64,
              resize_height=64, resize_width=64,
              crop=False, grayscale=False):
  image = imread(image_path, grayscale)
  return patch(np.array(image)/127.5 - 1)
  #return transform(image, input_height, input_width,
  #                 resize_height, resize_width, crop)

def get_image_old(image_path, input_height=64, input_width=64,
              resize_height=64, resize_width=64,
              crop=False, grayscale=False):
  image = imread(image_path, grayscale)
  return (np.array(image)/127.5 - 1)
  #return transform(image, input_height, input_width,
  #                 resize_height, resize_width, crop)

def patch(image):
  window_shape = (64, 64)
  B = view_as_windows(image, window_shape, step=32)
  return B

def imread(path, grayscale = False):
  return abs(ra.read_ra(os.path.join(path)))



data = glob(("./data/{0}/*.ra").format("train_img_slices"))
sample_files = data[0:16]
sample_patch = [get_image(sample_file) for sample_file in sample_files]
sample = np.reshape(sample_patch, (np.shape(sample_patch)[0] * np.shape(sample_patch)[1] * np.shape(sample_patch)[2], 64, 64))

#for i in range(len(sample_patch)):
#    b = [patch for a in sample_patch[i]]
#    sample.extend(b)
print("len sample_patch: ", np.shape(sample_patch))
print("len sample: ", np.shape(sample))

sample_patch = [get_image_old(sample_file) for sample_file in sample_files]
print("len sample old: ", np.shape(sample_patch))

