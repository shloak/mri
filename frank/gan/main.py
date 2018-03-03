import os
import scipy.misc
import numpy as np

from model import DCGAN
from utils import pp, visualize, to_json, show_all_variables

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 20, "Epoch to train [20]") # changed from  15
flags.DEFINE_integer("inner_iter", 100, "Inner Iteration")
flags.DEFINE_float("step_size", 0.001, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("learning_rate", 0.0001, "Learning rate of for adam [0.0002]")
flags.DEFINE_integer("batch_size", 1, "The size of batch images [64]")
flags.DEFINE_integer("sample_num", 4, "The size of sample images [4]")
flags.DEFINE_string("checkpoint_dir", "checkpoints", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("gen_dir", "gen", "Directory name to save the image samples [samples]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
FLAGS = flags.FLAGS

def main(_):
  pp.pprint(flags.FLAGS.__flags)

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)
  if not os.path.exists(FLAGS.gen_dir):
    os.makedirs(FLAGS.gen_dir)

  run_config = tf.ConfigProto()
  run_config.gpu_options.allow_growth=True

  with tf.Session(config=run_config) as sess:
    dcgan = DCGAN(sess,
                  batch_size=FLAGS.batch_size,
                  sample_num=FLAGS.sample_num,
                  checkpoint_dir=FLAGS.checkpoint_dir)

    show_all_variables()

    if FLAGS.train:
      dcgan.train(FLAGS)
    else:
      if not dcgan.load(FLAGS.checkpoint_dir)[0]:
        raise Exception("[!] Train a model first, then run test mode")

    dcgan.test(FLAGS)

    if FLAGS.visualize:
      OPTION = 0
      visualize(sess, dcgan, FLAGS, OPTION)

if __name__ == '__main__':
  tf.app.run()
