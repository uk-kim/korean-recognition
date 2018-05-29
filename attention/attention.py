'''
Reference
  DRAM : https://github.com/atulkum/paper_implementation/blob/master/MultipleObjectRecognitionWithVisualAttention.ipynb
  RAM : https://github.com/jtkim-kaist/ram_modified/blob/master/ram_modified.py
'''
import tensorflow as tf

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

    batch_size = 1#img.get_shape().as_list()[0]

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


def get_glimpse_feature(glimpse, w, b):
    # reshape glimpse tensor as [batch_size, height, width, g_depth]
    trsp_glimpse = tf.transpose(glimpse, [0, 2, 3, 1])

    conv1 = conv2d(trsp_glimpse, w['wg1'], b['bg1'])
    conv2 = conv2d(conv1, w['wg2'], b['bg2'])
    conv3 = conv2d(conv2, w['wg3'], b['bg3'])

    fc = tf.reshape(conv3, [-1, w['wgfc'].get_shape().as_list()[0]])
    feature = tf.nn.relu(tf.matmul(fc, w['wgfc']) + b['wgfc'])
    return feature


def glimpse_network(img, w, b, loc):
    # get input using the previous location
    glimpse_input = glimpse_sensor(img, loc)

    # the hidden units that process location & the input
    act_glimpse_hidden = get_glimpse_feature(glimpse_input, w, b)
    act_loc_hidden = tf.nn.relu(tf.matmul(loc, w['wlh']) + b['wlh'])

    # the hidden units that integrates the location & the glimpses
    glimpse_feature = tf.matmul(act_glimpse_hidden, w['wgh_gf']) + tf.matmul(act_loc_hidden, w['wlh_gf']) + b['wglh_gf']
    glimpse_feature = tf.nn.relu(glimpse_feature)

    return glimpse_feature


def context_network(img, w, b):
    img_g = tf.image.resize_images(img, patch_size, patch_size)
    img_g = tf.reshape(img_g, shape=[-1, patch_size, patch_size, 1])

    conv1 = conv2d(img_g, w['wc1'], b['bc1'])
    conv2 = conv2d(conv1, w['wc2'], b['bc2'])
    conv3 = conv2d(conv2, w['wc3'], b['bc3'])

    fc = tf.reshape(conv3, [-1, w['wc'].get_shape().as_list()[0]])
    fc = tf.add(tf.matmul(fc, w['wc']), b['bc'])
    return tf.nn.relu(fc)


def emission_network(output, w, b):
    # the next location is computed by the location network next of core-net(Level 2 RNN Cell)
    core_net_out = tf.stop_gradient(output)

    baseline = tf.sigmoid(tf.matmul(output, w['wbl']) + b['bbl'])

    # compute the next location, then impose noise
    if eye_centered:
        # add the last sampled glimpse location
        # TODO max(-1, min(1, u + N(output, sigma) + prevLoc)
        mean_loc = tf.maximum(-1.0, tf.minimum(1.0, tf.matmul(core_net_out, w['wh_nl'])))
    else:
        mean_loc = tf.matmul(core_net_out, w['wh_nl']) + b['bh_nl']
        mean_loc = tf.clip_by_value(mean_loc, -1, 1)

    # add noise
    sample_loc = tf.clip_by_value(mean_loc + tf.random_normal(mean_loc.get_shape(), 0, loc_sd), -1, 1)
    # don't propagate through the locations
    sample_loc = tf.stop_gradient(sample_loc)

    return mean_loc, sample_loc, baseline


def model(img, w, b):
    # initialize the location under uniform[-1, 1], for all example in the batch
    batch_size = 1  # img.get_shape().as_list()[0]
    mean_locs = []
    sampled_locs = []
    outputs = []
    baselines = []

    # context feature from origin image is initial state of the top core network layer.
    context_feature = context_network(img, w, b)

    rnn1 = tf.nn.rnn_cell.LSTMCell(lstm_size)
    rnn2 = tf.nn.rnn_cell.LSTMCell(lstm_size)

    h1 = tf.zeros([None, lstm_size])

    with tf.variable_scope('rnn2', reuse=False):
        h2, state2 = rnn2(h1, context_feature)
        # initialize the location under uniform[-1, 1], for all example in the batch
        mean_loc, sampled_loc, baseline = emission_network(h2, w, b)

    mean_locs.append(mean_loc)
    sampled_loc.append(sampled_loc)
    baselines.append(baseline)

    # initialize state of 1st rnn layer as zero values
    state1 = rnn1.zeros_state(None, tf.float32)

    '''
    이 부분에서 output을 계산하는 과정의 weight, bias의 파라미터를 초성, 중성, 종성마다 다르게 해야함
    즉, 초성의 경우 초성의 가짓수, 종성은 종성의 가짓수 등으로 해야하므로
    for 문을 T로 돌리는 것이 아니라 초성 따로, 중성 따로, 종성 따로 해서
    각각 n_glimpse_per_element 수만큼의 포문을 3번 반복해야함
    3번 반복하는 것을 할 때, weight를 리스트 형태로 반복해서 for 문으로 표현해도 될듯
    '''
    for t in range(T):
        glimpse = glimpse_network(img, w, b, sampled_loc)

        with tf.variable_scope('rnn1', reuse=(t != 0)):
            h1, state1 = rnn1(glimpse, state1)
            output = tf.sigmoid(tf.add(tf.matmul(h1, w['wo']), b['bo']))
        with tf.variable_scope('rnn2', reuse=True):
            h2, state2 = rnn2(h1, state2)
            mean_loc, sampled_loc, baseline = emission_network(h2, w, b)

        mean_locs.append(mean_loc)
        sampled_loc.append(sampled_loc)
        baselines.append(baseline)
        outputs.append(output)

    '''
    outputs : output list of 1st rnn that for decide agent's action(classification)
    mean_locs : predicted next location
    sampled_locs : random noise added location from mean_locs
    baselines : output list of 2nd rnn
    '''
    return outputs, mean_locs, sampled_locs, baselines


def calc_reward(outputs):
    '''
    reward를 계산할 시, 초성 중성 종성에 따라 차원의 수에 유의해야 함
    그리고 seq2seq 모델의 형태와 같이 시작과 끝을 나타내는 시그널(?)을 만드는 장치를 추가 하면 좋을듯
     --> 가: ㄱ + ㅏ  (초성 + 중성)
         감: ㄱ + ㅏ + ㅁ (초성 + 중성 + 종성)

         즉, 중성 다음에 종성이 올지 말지에 대해서는 중성 다음에 시퀀스 종료 시그널을 통해 파악하면 될것 같다.
         유념해서 반영하자.
    '''
    1
