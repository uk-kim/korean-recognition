import tensorflow as tf
import numpy as np

from attention.config import *


def conv2d(x, W, b, strides=1):
    out = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    out = tf.nn.bias_add(out, b)
    return tf.nn.relu(out)


def glimpse_network(x, w, b, loc):
    x_g = tf.image.extract_glimpse(x, tf.shape(patch_size, patch_size), loc)
    x_g = tf.reshape(x_g, shape=[-1, patch_size, patch_size, 1])

    conv1 = conv2d(x_g, w['wg1'], b['bg1'])
    conv2 = conv2d(conv1, w['wg2'], b['bg2'])
    conv3 = conv2d(conv2, w['wg3'], b['bg3'])

    fc1 = tf.reshape(conv3, [-1, w['wg'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, w['wg']), b['bg'])
    Gimage = tf.nn.relu(fc1)

    Gloc = tf.add(tf.matmul(loc, w['wgl']), b['bgl'])
    gn = tf.multiply(Gimage, Gloc)
    return gn


def context_network(x, w, b):
    x_g = tf.image.resize_images(x, patch_size, patch_size)
    x_g = tf.reshape(x_g, shape=[-1, patch_size, patch_size, 1])

    conv1 = conv2d(x_g, w['wc1'], b['bc1'])
    conv2 = conv2d(conv1, w['wc2'], b['bc2'])
    conv3 = conv2d(conv2, w['wc3'], b['bc3'])

    fc1 = tf.reshape(conv3, [-1, w['wc'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, w['wc']), b['bc'])
    return tf.nn.relu(fc1)