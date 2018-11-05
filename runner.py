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
#%matplotlib inline
def show_images(images, cols = 1):
    n_images = len(images)
    fig = plt.figure()
    for n, image in enumerate(images):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        plt.imshow(image, cmap='gray')
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()
    


# returns [0, 1] normalized combination of the images (320, 256, 2) w factors lambda[i]
def combine_images(images, lambdas):
    res = np.zeros((320, 256, 2))
    for im, lam in zip(images, lambdas):
        minv, maxv = np.amin(im[:, :, 0]), np.amax(im[:, :, 0])
        im[:, :, 0] = (im[:, :, 0] - minv) / (maxv - minv)
        minv, maxv = np.amin(im[:, :, 1]), np.amax(im[:, :, 1])
        im[:, :, 1] = (im[:, :, 1] - minv) / (maxv - minv)
        res += lam * im
    #plt.imshow(res[:, :, 0].reshape((320, 256)), cmap='gray')
    #plt.show()
    return res

def get_loss(images, original):
    orig_normal = np.zeros((320, 256, 2))
    minv, maxv = np.amin(original[:, :, 0]), np.amax(original[:, :, 0])
    orig_normal[:, :, 0] = (original[:, :, 0] - minv) / (maxv - minv)
    minv, maxv = np.amin(original[:, :, 1]), np.amax(original[:, :, 1])
    orig_normal[:, :, 1] = (original[:, :, 1] - minv) / (maxv - minv)

    step = 0.05
    errors_comb, errors_real, steps = [], [], []    
    for i in range(int(1/step) + 1):
        comb = combine_images(images, [i * step, 1 - (i * step)])
        errors_comb.append(np.sum((orig_normal - comb) ** 2))        
        errors_real.append(np.sum((orig_normal[:, :, 0] - comb[:, :, 0]) ** 2))   
        steps.append(step * i)
    return (steps, errors_comb, errors_real)



height = 320  # 320, 218
width = 256   # 256, 178

#flags

flags = tf.app.flags
flags.DEFINE_integer("epoch", 30, "Epoch to train [20]") # changed to 20 from  25
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 4, "The size of batch images [64]")
flags.DEFINE_integer("input_height", height, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", width, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", height, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", width, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("dataset", "train_img_slices", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "2chanv2", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples_2chan", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
FLAGS = flags.FLAGS

pp.pprint(flags.FLAGS.__flags)

if FLAGS.input_width is None:
    FLAGS.input_width = FLAGS.input_height
if FLAGS.output_width is None:
    FLAGS.output_width = FLAGS.output_height

if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth=True

#data = glob("./celebA/*.jpg")[3005:3009] #face
data = glob("./data/valid_img_slices/*.ra")         #mri
#np.random.shuffle(data)

sample_files = data[210:214]
#sample = [get_image_old1(sample_file, True) for sample_file in sample_files] #face
#sample = [cv2.resize(s, dsize=(256, 320)) for s in sample]                   #face
sample = [get_image_old2(sample_file) for sample_file in sample_files]        #mri
print(sample_files)

plt.imshow(np.sqrt(sample[0][:, :, 0]**2 + sample[0][:, :, 1]**2), cmap ='gray')
#plt.show()
plt.imshow(sample[0][:, :, 0], cmap ='gray')
plt.show()

#v = np.reshape(sample, (4, height*width))

v = np.reshape(sample, (4, height*width*2))[3:4]

# number of features per image (pixels including all channels)
print('shape of v: {}'.format(v.shape))


n = v.shape[1]

# [2x, 3x, 4x, 5x, 6x, 7x, 8x] DELL COMP
masks = glob("./masks/masks/*.ra")

img = ra.read_ra(masks[6]).T
mag = abs(img)
mag = (mag > 0.5)*1.
the_mask = np.fft.fftshift(mag)

###masks = glob("./masks/masks/12_*")
###the_mask = np.load(masks[0])

print(np.shape(the_mask))
m = sum(sum(the_mask))
print(m)

comb_sample = sample[0][:, :, 0] + 1j*sample[0][:, :, 1]

kspace = np.fft.fft2(comb_sample) / np.sqrt(height*width)
kspace = np.multiply(kspace, the_mask) # if subsample

inv = np.fft.ifft2(kspace)
print(np.shape(inv))
plt.imshow(np.sqrt(np.real(inv)**2 + np.imag(inv)**2), cmap='gray')
#plt.show()
plt.imshow(np.concatenate([np.real(inv), np.imag(inv)]), cmap='gray')
#plt.show()


kspace = np.reshape(kspace, (-1))
m_ = np.shape(kspace)[0]
print(m_)

possible_ms = [10000, 100, 500, 1000, 2000, 5000, 6000]
num_iter = 4000 #2000
compressed_images = []

plt.imshow(np.sqrt(sample[0][:, :, 0]**2 + sample[0][:, :, 1]**2), cmap='gray')
#plt.show()

ims = []

with tf.Session() as sess:

    #print(tf.global_variables())

    dcgan = DCGAN(
        sess,
        input_width=FLAGS.input_width,
        input_height=FLAGS.input_height,
        output_width=FLAGS.output_width,
        output_height=FLAGS.output_height,
        batch_size=FLAGS.batch_size,
        sample_num=FLAGS.batch_size,
        dataset_name=FLAGS.dataset,
        input_fname_pattern=FLAGS.input_fname_pattern,
        crop=FLAGS.crop,
        checkpoint_dir=FLAGS.checkpoint_dir,
        sample_dir=FLAGS.sample_dir)



    if not dcgan.load(FLAGS.checkpoint_dir)[0]:
        raise Exception("[!] Train a model first, then run test mode")

    dcgan.z.initializer.run()

    #for m in possible_ms:
    for i in range(1):
    
        
        y_placeholder2 = tf.placeholder(tf.complex64,[m_])
        mask= tf.cast(the_mask, dtype=tf.complex64)
        ####
        X = tf.Variable(tf.random_uniform([320, 256, 2]), name='Ximage')
        X2 = tf.squeeze(tf.cast(X, dtype=tf.complex64))
        X_fft = X2[:, :, 0] + 1j*X2[:, :, 1]
        X_fft = tf.fft2d(X_fft) 
        X_fft = tf.multiply(X_fft, mask) / np.sqrt(height*width) # if subsample
        X_fft = tf.cast(tf.reshape(X_fft, [-1]), dtype=tf.complex64) 

        x_loss = (tf.reduce_sum(tf.abs(X_fft - y_placeholder2)**2))
        lp_loss_x = tf.norm(X, ord=2)
        tv_loss_x = tf.image.total_variation(X)
        x_combined_loss = x_loss + 0 * lp_loss_x + 0.005 * tv_loss_x   
        ####


        chan_2 = tf.squeeze(tf.cast(dcgan.G[0],dtype=tf.complex64))
        chan_2 = chan_2[:, :, 0] + 1j*chan_2[:, :, 1]
        g_kspace = tf.fft2d(chan_2)
        g_kspace = tf.multiply(g_kspace, mask) # if subsample
        g_kspace = tf.reshape(g_kspace, [-1]) / np.sqrt(height*width)

        z_loss = tf.reduce_sum(tf.abs(g_kspace - y_placeholder2)**2)
        
        lp_loss_z = (tf.norm(tf.squeeze(dcgan.G[0]), ord=2))
        tv_loss_z = tf.image.total_variation(tf.squeeze(dcgan.G[0]))
        z_combined_loss = z_loss + 10 * lp_loss_z + 0.0001 * tv_loss_z 

        #z_optim = tf.train.AdamOptimizer(learning_rate=0.05).minimize(our_loss + lp * lp_loss + tv * tv_loss, var_list=dcgan.z)  
        comb_optim = tf.train.AdamOptimizer(learning_rate=0.05).minimize(1 * z_combined_loss + 1 * x_combined_loss, var_list=[dcgan.z, X])

        opt_initializers   = [var.initializer for var in tf.global_variables() if 'Adam' in var.name]
        beta1_initializers = [var.initializer for var in tf.global_variables() if 'beta1_power' in var.name]
        beta2_initializers = [var.initializer for var in tf.global_variables() if 'beta2_power' in var.name]
        
        x_init= [var.initializer for var in tf.global_variables() if 'Ximage' in var.name]

        sess.run(opt_initializers)
        sess.run(beta1_initializers)
        sess.run(beta2_initializers)
        sess.run(x_init)
        
        count = 0
        errors = []
        #lps = []
        #tvs = []
        aas = []

        for i in range(num_iter): 
            #a, closs, b = sess.run([comb_optim, our_loss, dcgan.G],feed_dict={y_placeholder2: kspace})
            a, zloss, xloss, b = sess.run([comb_optim, z_combined_loss, x_combined_loss, dcgan.G],feed_dict={y_placeholder2: kspace})
            count += 1
            errors.append(zloss)
            #lps.append(lp)
            #tvs.append(tv)
            aas.append(dcgan.z.eval())
            if np.mod(count, 100) == 1:   
                #if count < 800:
                #    noise = tf.random_normal([4, 1000], 0.0, 1.0)
                #    dcgan.z += noise
                print('iteration {}'.format(count))
                print('x:', xloss, ' z:', zloss, 'comb:', 5*xloss+zloss)
            if np.mod(count, 1000) == 1:
                ims.append(b[0])
                x_curr = X.eval()
                #combine_images([x_curr, ims[-1]], [1, 0])
                #combine_images([x_curr, ims[-1]], [0.75, 0.25])
                #combine_images([x_curr, ims[-1]], [0.5, 0.5])
                #combine_images([x_curr, ims[-1]], [0.25, 0.75])
                #combine_images([x_curr, ims[-1]], [0, 1])
                steps, totals, reals = get_loss([x_curr, ims[-1]], sample[0])
                plt.plot(steps, reals)
                plt.title('REAL error vs fraction of non-gan')
                plt.show()
                plt.plot(steps, totals)
                plt.title('TOTAL error vs fraction of non-gan')
                plt.show()
        res = b[0].reshape((height, width, 2))
