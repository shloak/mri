from __future__ import division
import os
import time
import math
import random
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

import imageio
imageio.plugins.ffmpeg.download()

from ops import *
from utils import *

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

class DCGAN(object):
  def __init__(self, sess,
               batch_size=1, sample_num=64,
               z_dim=20, gf_dim=32, df_dim=32,
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
    self.sample_num = sample_num

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

    self.data = glob("./data/train_img_slices/*.ra")
    random.shuffle(self.data)
    
    self.height, self.width = imread(self.data[0]).shape

    self.build_model()

  def build_model(self):

    image_dims = [self.height, self.width, 1]
    self.inputs = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='real_images') 

    inputs = self.inputs

    self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
    self.z_sum = histogram_summary("z", self.z)

    self.G                  = self.generator(self.z, reuse=False)
    self.D, self.D_logits   = self.discriminator(inputs, reuse=False)
    self.sampler            = self.sampler(self.z)
    self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)
    
    self.d_sum = histogram_summary("d", self.D)
    self.d__sum = histogram_summary("d_", self.D_)
    self.G_sum = image_summary("G", self.G)

    def sigmoid_cross_entropy_with_logits(x, y):
      try:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
      except:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

    self.d_loss_real = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
    self.d_loss_fake = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
    self.g_loss_fake = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

    self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
    self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
                          
    self.d_loss = self.d_loss_real + self.d_loss_fake
    self.g_loss = self.g_loss_fake

    self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
    self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

    t_vars = tf.trainable_variables()

    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]

    self.saver = tf.train.Saver()

  def train(self, config):
    d_optim = tf.train.GradientDescentOptimizer(config.learning_rate) \
                      .minimize(self.d_loss, var_list=self.d_vars)
    g_optim = tf.train.GradientDescentOptimizer(config.learning_rate) \
                      .minimize(self.g_loss, var_list=self.g_vars)
    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()

    self.g_sum = merge_summary([self.z_sum, self.d__sum,
      self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
    self.d_sum = merge_summary(
        [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
    self.writer = SummaryWriter("./logs", self.sess.graph)

    sample_z = np.random.uniform(-1, 1, size=(self.sample_num , self.z_dim))
  
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
        batch = [imread(d) for d in batch_files]

        batch_images = np.array(batch).astype(np.float32)[:, :, :, None]

        batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]).astype(np.float32)

        # Update D network
        self.sess.run(d_optim,
                        feed_dict={ self.inputs: batch_images, self.z: batch_z })

        # Update G network
        self.sess.run(g_optim,
                      feed_dict={ self.inputs: batch_images, self.z: batch_z })

        errD = self.d_loss.eval({self.z: batch_z, self.inputs: batch_images })
        errG = self.g_loss.eval({self.z: batch_z, self.inputs: batch_images })

        counter += 1
        print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
          % (epoch, idx, batch_idxs,
            time.time() - start_time, errD, errG))
        
        if counter % 25 == 0:
          samples = self.sess.run(self.sampler, feed_dict={self.z: sample_z})
          save_images(samples, image_manifold_size(samples.shape[0]),
                      './{}/{}.png'.format(config.sample_dir, addZeros(epoch, idx, batch_idxs)))

        if counter % 500 == 2:
          self.save(config.checkpoint_dir, counter)

  def discriminator(self, image, reuse=False):
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()

      h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
      h1 = lrelu(self.d_bn1(conv2d(h0, 2 * self.df_dim, name='d_h1_conv')))
      h2 = lrelu(self.d_bn2(conv2d(h1, 4 * self.df_dim, name='d_h2_conv')))
      h3 = lrelu(self.d_bn3(conv2d(h2, 8 * self.df_dim, name='d_h3_conv')))
      h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin')

      return tf.nn.sigmoid(h4), h4

  def generator(self, z, reuse=False):
    with tf.variable_scope("generator") as scope:
      if reuse:
        scope.reuse_variables()
        
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

  def sampler(self, z):
    with tf.variable_scope("generator") as scope:
      scope.reuse_variables()

      s_h, s_w = self.height, self.width
      s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
      s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
      s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
      s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

      # project `z` and reshape
      h0 = tf.reshape(linear(z, 8 * self.gf_dim * s_h16 * s_w16, 'g_h0_lin'),
                      [-1, s_h16, s_w16, 8 * self.gf_dim])
      h0 = tf.nn.relu(self.g_bn0(h0, train=False))

      h1 = deconv2d(h0, [self.sample_num, s_h8, s_w8, 4 * self.gf_dim], name='g_h1')
      h1 = tf.nn.relu(self.g_bn1(h1, train=False))

      h2 = deconv2d(h1, [self.sample_num, s_h4, s_w4, 2 * self.gf_dim], name='g_h2')
      h2 = tf.nn.relu(self.g_bn2(h2, train=False))

      h3 = deconv2d(h2, [self.sample_num, s_h2, s_w2, self.gf_dim], name='g_h3')
      h3 = tf.nn.relu(self.g_bn3(h3, train=False))

      h4 = deconv2d(h3, [self.sample_num, s_h, s_w, 1], name='g_h4')

      return tf.nn.tanh(h4)
      
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

  def test(self, config):
    
    z_optim = tf.train.GradientDescentOptimizer(config.step_size) \
                      .minimize(self.g_loss_real, var_list=self.z_var)

    # opt_initializers   = [var.initializer for var in tf.global_variables() if 'Adam' in var.name]
        
    sess.run(opt_initializers)
    
    start_time = time.time()

    for epoch in xrange(config.epoch):
      batch_idxs = len(self.data) // config.batch_size
      batch_files = self.data[:config.batch_size]
      batch = [imread(d) for d in batch_files]
      batch_images = np.array(batch).astype(np.float32)[:, :, :, None]

    for i in range(num_iter): 
      print('iteration {}'.format(count))
      a, closs, b = sess.run(z_optim, our_loss, feed_dict={y_placeholder: v})
