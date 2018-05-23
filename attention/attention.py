'''
Reference
  DRAM : https://github.com/atulkum/paper_implementation/blob/master/MultipleObjectRecognitionWithVisualAttention.ipynb
  RAM : https://github.com/jtkim-kaist/ram_modified/blob/master/ram_modified.py
'''

import tensorflow as tf
import numpy as np

from attention.config import *


def conv2d(x, W, b, strides=1):
    out = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    out = tf.nn.bias_add(out, b)
    return tf.nn.relu(out)


def glimpse_sensor(img, norm_loc):
    # norm_loc coordinates are between -1 and 1
    loc = tf.round(((norm_loc + 1) / 2.0) * img_sz)
    loc = tf.cast(loc, tf.int32)

    img = tf.reshape(img, (-1, img_sz, img_sz, channels))

    batch_size = img.get_shape().as_list()[0]

    # process each image individually
    zooms = []
    for k in range(batch_size):
        img_zooms = []
        one_img = img[k, :, :, :]
        max_radius = patch_size * (2 ** (g_depth - 1))
        offset = 2 * max_radius

        # pad image with zeros
        one_img = tf.image.pad_to_bounding_box(image=one_img,
                                               offset_height=offset,
                                               offset_width=offset,
                                               target_height=max_radius*4+img_sz,
                                               target_width=max_radius*4+img_sz)

        for i in range(g_depth):
            r = int(patch_size * (2 ** i))

            d_raw = 2 * r
            d = tf.constant(d_raw, shape=[1])
            d = tf.tile(d, [2])
            loc_k = loc[k, :]
            adjusted_loc = offset + loc_k - r
            one_img2 = tf.reshape(one_img, (one_img.get_shape()[0].value,
                                            one_img.get_shape()[1].value))

            # crop image to (d x d)
            zoom = tf.slice(one_img2, adjusted_loc, d)

            # resize cropped image to (sensorBandwidth x sensorBandwidth)
            zoom = tf.image.resize_bilinear(tf.reshape(zoom, (1, d_raw, d_raw, 1)),
                                            (sensor_bandwidth, sensor_bandwidth))
            zoom = tf.reshape(zoom, (sensor_bandwidth, sensor_bandwidth))
            img_zooms.append(zoom)

        zooms.append(tf.stack(img_zooms))

    zooms = tf.stack(zooms)
    return zooms



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


def model(x, w, b):
    y = [0] * T
    loc = [0] * (T+1)

    rnn1 = tf.nn.rnn_cell.LSTMCell(lstm_size)
    rnn2 = tf.nn.rnn_cell.LSTMCell(lstm_size)

    h1 = tf.zeros([None, lstm_size])
    state2 = context_network(x, w, b)

    with tf.variable_scope('rnn2', reuse=False):
        h2, state2 = rnn2(h1, state2)
        loc[0] = tf.add(tf.matmul(h2, w['w1']), b['b1'])

    state1 = rnn1.zero_state(None, tf.float32)
    for t in range(T):
        gn = glimpse_network(x, w, b, loc[t])
        with tf.variable_scope('rnn1', reuse=(t != 0)):
            h1, state1 = rnn1(gn, state1)
            y[t] = tf.add(tf.matmul(h1, w['wo']), b['bo'])
        with tf.variable_scope('rnn2', reuse=True):
            h2, state2 = rnn2(h1, state2)
            loc[t+1] = tf.add(tf.matmul(h2, w['w1']), b['b1'])

    return y, loc


