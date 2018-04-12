from __future__ import division
import os
import time
import math
import random
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

class DCAM(object):
  def __init__(self, sess,
               batch_size=1,
               z_dim=1000, gf_dim=32, df_dim=32,
               checkpoint_dir=None):
    """

    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      z_dim: (optional) Dimension of dim for z. [100]
      gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
      df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
    """
    self.sess = sess

    self.batch_size = batch_size

    self.z_dim = z_dim

    self.gf_dim = gf_dim
    self.df_dim = df_dim

    self.d_bn1 = batch_norm(name='d_bn1')
    self.d_bn2 = batch_norm(name='d_bn2')
    self.d_bn3 = batch_norm(name='d_bn3')

    self.g_bn0 = batch_norm(name='g_bn0')
    self.g_bn1 = batch_norm(name='g_bn1')
    self.g_bn2 = batch_norm(name='g_bn2')
    self.g_bn3 = batch_norm(name='g_bn3')

    self.checkpoint_dir = checkpoint_dir

    #self.data = glob("../data/train_img_slices/*.ra")
    self.data = glob("../img_align_celeba/*.jpg")[:3000]
    random.shuffle(self.data)
    
    self.height, self.width = imread(self.data[0], True).shape

    self.build_model()

  def build_model(self):

    image_dims = [self.height, self.width, 1]
    self.inputs = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='real_images') 

    inputs = self.inputs

    self.z = tf.Variable(tf.random_normal([self.batch_size, self.z_dim],
                                          stddev=1e-5, dtype=tf.float32),
                         name='z')
    self.z_sum = histogram_summary("z", self.z)

    self.Gz = self.generator(self.z)
    
    self.D, self.D_logits = self.discriminator(inputs)
    self.D_, self.D_logits_ = self.discriminator(self.Gz, reuse=True)
    self.d_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
    self.g_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))
    # TODO: create g optim and d optim, figure out when to run them in train

    self.loss = tf.reduce_mean((self.Gz - inputs)**2)
    
    l1_regularizer = tf.contrib.layers.l1_regularizer(
       scale=0.005, scope=None)
    self.L1 = tf.contrib.layers.apply_regularization(l1_regularizer, [self.z])
    
    self.total_loss = self.loss + self.L1

    t_vars = tf.trainable_variables()
    

    self.g_vars = [var for var in t_vars if 'g_' in var.name]
    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    
    self.saver = tf.train.Saver()

  def train(self, config):
    z_optim = tf.train.GradientDescentOptimizer(config.learning_rate_z) \
                      .minimize(self.loss, var_list=self.z)
    g_optim = tf.train.GradientDescentOptimizer(config.learning_rate_g) \
                      .minimize(self.loss, var_list=self.g_vars)
    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()

    counter = 1
    start_time = time.time()
    could_load, checkpoint_counter = self.load(self.checkpoint_dir)
    if could_load:
      counter = checkpoint_counter
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    for epoch in xrange(config.epoch):
      batch_idxs = len(self.data) // config.batch_size

      for idx in xrange(batch_idxs):
        batch_files = self.data[idx*(config.batch_size):(idx+1)*(config.batch_size)]
        batch = [imread(d, True) for d in batch_files]

        batch_images = np.expand_dims(np.array(batch).astype(np.float32), axis=-1)

        # Update z
        self.sess.run(self.z.initializer)
        for it in xrange(config.z_inner_iter):
          self.sess.run(z_optim,
                        feed_dict={ self.inputs: batch_images })
          err = self.loss.eval({ self.inputs: batch_images })
          print("Epoch: [%2d] z_iter: [%2d], [%4d/%4d], loss: %.8f" \
                % (epoch, it, idx, batch_idxs, err))

        # Update G network
        for it in xrange(config.g_inner_iter):
          self.sess.run(g_optim,
                        feed_dict={ self.inputs: batch_images })
          err = self.loss.eval({ self.inputs: batch_images })
          print("Epoch: [%2d] g_iter: [%2d], [%4d/%4d], loss: %.8f" \
                % (epoch, it, idx, batch_idxs, err))

        counter += 1        
        if counter % 50 == 0:
          Gz = self.sess.run(self.Gz)
          save_images(Gz, image_manifold_size(Gz.shape[0]),
                      './{}/{}.png'.format(config.sample_dir, addZeros(epoch, idx, batch_idxs)))

        if counter % 250 == 2:
          self.save(config.checkpoint_dir, counter)

  def generator(self, z):
    with tf.variable_scope("generator") as scope:
        
      s_h, s_w = self.height, self.width
      s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
      s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
      s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
      s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

      # project `z` and reshape
      self.z_, self.h0_w, self.h0_b = linear(z, 8 * self.gf_dim * s_h16 * s_w16,
                                             'g_h0_lin', with_w=True)

      self.h0 = tf.reshape(self.z_, [-1, s_h16, s_w16, 8 * self.gf_dim])
      h0 = tf.nn.relu(self.g_bn0(self.h0))

      self.h1, self.h1_w, self.h1_b = deconv2d(
          h0, [self.batch_size, s_h8, s_w8, 4 * self.gf_dim], name='g_h1', with_w=True)
      h1 = tf.nn.relu(self.g_bn1(self.h1))

      h2, self.h2_w, self.h2_b = deconv2d(
          h1, [self.batch_size, s_h4, s_w4, 2 * self.gf_dim], name='g_h2', with_w=True)
      h2 = tf.nn.relu(self.g_bn2(h2))

      h3, self.h3_w, self.h3_b = deconv2d(
          h2, [self.batch_size, s_h2, s_w2, self.gf_dim], name='g_h3', with_w=True)
      h3 = tf.nn.relu(self.g_bn3(h3))

      h4, self.h4_w, self.h4_b = deconv2d(
          h3, [self.batch_size, s_h, s_w, 1], name='g_h4', with_w=True)

      return tf.nn.tanh(h4)

  def discriminator(self, image, reuse=False):
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()
        
      h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
      h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
      h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
      h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
      h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin')

      return tf.nn.sigmoid(h4), h4  
    
  @property
  def model_dir(self):
    return "{}_{}_{}".format(self.batch_size, self.height, self.width)
      
  def save(self, checkpoint_dir, step):
    model_name = "DCGAN.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)
    zsave = np.array(self.z.eval())
    np.save(('./{}/{}').format(checkpoint_dir, str(step)) , zsave)

  def load(self, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0
