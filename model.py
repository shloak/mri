from __future__ import division
import os
import time
import math
import cv2
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
  def __init__(self, sess, input_height=64, input_width=80, crop=False,
         batch_size=64, sample_num = 64, output_height=64, output_width=80,
         y_dim=None, z_dim=1000, gf_dim=64, df_dim=64, # change z_dim * 20
         gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='train_img_slices', 
         input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None):
    """

    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      y_dim: (optional) Dimension of dim for y. [None]
      z_dim: (optional) Dimension of dim for Z. [100]
      gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
      df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
      gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
      dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
      c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
    """
    self.sess = sess
    self.crop = crop

    self.batch_size = batch_size
    self.sample_num = sample_num

    self.input_height = input_height
    self.input_width = input_width
    self.output_height = output_height
    self.output_width = output_width

    self.y_dim = y_dim
    self.z_dim = z_dim

    self.gf_dim = gf_dim
    self.df_dim = df_dim

    self.gfc_dim = gfc_dim
    self.dfc_dim = dfc_dim

    # batch normalization : deals with poor initialization helps gradient flow
    self.d_bn1 = batch_norm(name='d_bn1')
    self.d_bn2 = batch_norm(name='d_bn2')

    if not self.y_dim:
      self.d_bn3 = batch_norm(name='d_bn3')
      #self.d_bn4 = batch_norm(name='d_bn4') #new
      #self.d_bn5 = batch_norm(name='d_bn5') #new

    self.g_bn0 = batch_norm(name='g_bn0')
    self.g_bn1 = batch_norm(name='g_bn1')
    self.g_bn2 = batch_norm(name='g_bn2')

    if not self.y_dim:
      self.g_bn3 = batch_norm(name='g_bn3')
      #self.g_bn4 = batch_norm(name='g_bn4') #new
      #self.g_bn5 = batch_norm(name='g_bn5') #new

    self.dataset_name = dataset_name
    self.input_fname_pattern = input_fname_pattern
    self.checkpoint_dir = checkpoint_dir

    if self.dataset_name == 'mnist':
      print('mnist')  
    else:
      #self.data = glob("./celebA/*.jpg")[:3000]
      self.data = glob("./data/train_img_slices/*.ra")
      random.shuffle(self.data)
      imreadImg = get_image_old2(self.data[0]);
      self.c_dim = 2#1

    self.grayscale = False#(self.c_dim == 1)

    self.build_model() 
    
  def build_model(self):
    if self.y_dim:
      self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')
    else:
      self.y = None

    if self.crop:
      image_dims = [self.output_height, self.output_width, self.c_dim]
    else:
      image_dims = [self.input_height, self.input_width, self.c_dim]

    self.inputs = tf.placeholder(
      tf.float32, [self.batch_size] + image_dims, name='real_images') 

    inputs = self.inputs

    self.the_mask = tf.placeholder(tf.complex64, [320, 256])
    self.mask_files = glob("./masks/gen_masks/6_*")
    self.mask_files = [np.fft.fftshift(np.load(m)) for m in self.mask_files]
    
    self.mask = random.choice(self.mask_files)
    self.z = tf.placeholder(
      tf.float32, [None, self.z_dim], name='z')
    self.z_sum = histogram_summary("z", self.z)

    self.G                  = self.generator(self.z, self.y)
    self.D, self.D_logits   = self.discriminator(inputs, self.the_mask, self.y, reuse=False)
    self.sampler            = self.sampler(self.z, self.y)
    self.D_, self.D_logits_ = self.discriminator(self.G, self.the_mask, self.y, reuse=True)
    
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
    self.g_loss = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

    self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
    self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
                          
    self.d_loss = self.d_loss_real + self.d_loss_fake

    self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
    self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

    t_vars = tf.trainable_variables()

    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]

    self.saver = tf.train.Saver()

  def train(self, config):
    d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.d_loss, var_list=self.d_vars)
    g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
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

    #sample_z = np.random.normal(0, 1e-5, size=(self.sample_num , self.z_dim))
    sample_z = np.random.uniform(-1, 1, size=(self.sample_num , self.z_dim))
    
    if config.dataset == 'mnist':
      print('mnist')  
    else:
      sample_files = self.data[0:(self.sample_num)]
      sample = [get_image_old2(d) for d in sample_files]
      #sample = [get_image_old1(d, True) for  d in sample_files]
      #sample = [cv2.resize(s, dsize=(256, 320)) for s in sample]
            
    if (self.grayscale):
        sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
    else:
        sample_inputs = np.array(sample).astype(np.float32)
  
    counter = 1
    countD = 1
    start_time = time.time()
    could_load, checkpoint_counter = self.load(self.checkpoint_dir)
    if could_load:
      counter = checkpoint_counter
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    for epoch in xrange(config.epoch):
      if config.dataset == 'mnist':
        print('mnist')    
      else:      

        batch_idxs = min(len(self.data), config.train_size) // config.batch_size # changed //16

      for idxs in xrange(0, batch_idxs):
        idx = np.mod(idxs, batch_idxs)
        if config.dataset == 'mnist':
          print('mnist')  
        else:
          batch_files = self.data[idx*(config.batch_size):(idx+1)*(config.batch_size)]
          batch = [get_image_old2(d) for d in batch_files]    
          #batch = [get_image_old1(d, True) for d in batch_files]
          #batch = [cv2.resize(b, dsize=(256, 320)) for b in batch]

          if self.grayscale:
            batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
          else:
            batch_images = np.array(batch).astype(np.float32)

        batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
              .astype(np.float32)
        self.zbatch = batch_z

        if config.dataset == 'mnist':
          print('mnist')  

        else:
          # Update D network
          if np.mod(countD, 7) == 0:
            self.mask = random.choice(self.mask_files)
          _ = self.sess.run([d_optim],
            feed_dict={ self.inputs: batch_images, self.the_mask: self.mask, self.z: batch_z})
          #self.writer.add_summary(summary_str, counter)
          countD += 1

          if np.mod(countD, 1) == 0:      
            # Update G network
            _ = self.sess.run([g_optim],
              feed_dict={ self.z: batch_z, self.the_mask: self.mask })
            #self.writer.add_summary(summary_str, counter)

            # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
            _ = self.sess.run([g_optim],
              feed_dict={ self.z: batch_z, self.the_mask: self.mask })
            #self.writer.add_summary(summary_str, counter)
          
            errD_fake = self.d_loss_fake.eval({ self.z: batch_z, self.the_mask: self.mask })
            errD_real = self.d_loss_real.eval({ self.inputs: batch_images, self.the_mask: self.mask })
            errG = self.g_loss.eval({self.z: batch_z, self.the_mask: self.mask})

            counter += 1
            print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
              % (epoch, idx, batch_idxs,
                time.time() - start_time, errD_fake+errD_real, errG))

        if np.mod(countD, 700) == 1: #700
          if config.dataset == 'mnist':
            print('mnist')
          else:
            #try:
            samples, d_loss, g_loss = self.sess.run(
                [self.sampler, self.d_loss, self.g_loss],
                  feed_dict={
                    self.z: batch_z, #sample_z
                    self.inputs: batch_images, #sample_inputs
                    self.the_mask: self.mask
                  },
            )
            save_images(samples, image_manifold_size(samples.shape[0]),
                  './{}/{}.png'.format(config.sample_dir, addZeros(0, countD)))
            print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 
            #except:
            #  print("one pic error!...")

        if np.mod(countD, 4000) == 2: #4000
          self.save(config.checkpoint_dir, counter)

  def discriminator(self, image, mask, y=None, reuse=False):
    c_image = tf.cast(tf.squeeze(image), dtype=tf.complex64)
    c_image = tf.stack([c_image[0, :, :, 0] + 1j*c_image[0, :, :, 1], c_image[1, :, :, 0] + 1j*c_image[1, :, :, 1], c_image[2, :, :, 0] + 1j*c_image[2, :, :, 1], c_image[3, :, :, 0] + 1j*c_image[3, :, :, 1]])
    #c_image = [img[:, :, 0] + 1j*im[:, :, 1] for im in c_image] # need to convert to 1 channel, fft, convert to 2 channel, discrim
    ffts = tf.fft2d(c_image)
    masked = tf.multiply(ffts, mask)
    ###iffts = tf.real(tf.ifft2d(masked))
    ###ifft_f = tf.cast(iffts, dtype=tf.float32)
    ###images = tf.expand_dims(ifft_f, -1)
    iffts = tf.ifft2d(masked)
    images = tf.cast(tf.stack([convert(iffts[0, :, :]), convert(iffts[1, :, :]), convert(iffts[2, :, :]), convert(iffts[3, :, :])]), dtype=tf.float32)
    with tf.variable_scope("discriminator") as scope:
      
      if reuse:
        scope.reuse_variables()

      if not self.y_dim:
        #h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
        h0 = lrelu(conv2d(images, self.df_dim, name='d_h0_conv'))
        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
        h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
        h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
        h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin')

        return tf.nn.sigmoid(h4), h4
      else:
        print('mnist')    

  def generator(self, z, y=None):
    with tf.variable_scope("generator") as scope:
      if not self.y_dim:
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        # project `z` and reshape
        self.z_, self.h0_w, self.h0_b = linear(
            z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True)

        self.h0 = tf.reshape(
            self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
        h0 = tf.nn.relu(self.g_bn0(self.h0))

        self.h1, self.h1_w, self.h1_b = deconv2d(
                h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1', with_w=True)
        h1 = tf.nn.relu(self.g_bn1(self.h1))

        h2, self.h2_w, self.h2_b = deconv2d(
            h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2', with_w=True)
        h2 = tf.nn.relu(self.g_bn2(h2))

        h3, self.h3_w, self.h3_b = deconv2d(
            h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3', with_w=True)
        h3 = tf.nn.relu(self.g_bn3(h3))

        h4, self.h4_w, self.h4_b = deconv2d(
            h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)

        return tf.nn.tanh(h4)
      else:
        print('mnist')

  def sampler(self, z, y=None):
    with tf.variable_scope("generator") as scope:
      scope.reuse_variables()

      if not self.y_dim:
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        # project `z` and reshape
        h0 = tf.reshape(
            linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin'),
            [-1, s_h16, s_w16, self.gf_dim * 8])
        h0 = tf.nn.relu(self.g_bn0(h0, train=False))

        h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1')
        h1 = tf.nn.relu(self.g_bn1(h1, train=False))

        h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2')
        h2 = tf.nn.relu(self.g_bn2(h2, train=False))

        h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3')
        h3 = tf.nn.relu(self.g_bn3(h3, train=False))

        h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4')

        return tf.nn.tanh(h4)
      else:
        print('mnist')      

  def load_mnist(self):
    data_dir = os.path.join("./data", self.dataset_name)
    
    fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.float)

    trY = np.asarray(trY)
    teY = np.asarray(teY)
    
    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)
    
    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)
    
    y_vec = np.zeros((len(y), self.y_dim), dtype=np.float)
    for i, label in enumerate(y):
      y_vec[i,y[i]] = 1.0
    
    return X/255.,y_vec

  @property
  def model_dir(self):
    return "{}_{}_{}_{}".format(
        self.dataset_name, self.batch_size,
        self.output_height, self.output_width)
      
  def save(self, checkpoint_dir, step):
    model_name = "DCGAN.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)
    
    zsave = np.array(self.zbatch)
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
