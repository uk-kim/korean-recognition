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


def weight_variable(shape, name=None):
    # initial = tf.truncated_normal(shape, stddev=0.1)
    # return tf.Variable(initial, name=name, dtype=tf.float32)
    return tf.get_variable(name=name, shape=shape, dtype=tf.float32)


def bias_variable(shape, name=None):
    # tf.get_variable('contextNet_bias_conv1', [16], tf.float32),
    # initial = tf.constant(0.0, shape=shape, dtype=tf.float32)
    # return tf.Variable(initial, name=name)
    return tf.get_variable(name, shape=shape, dtype=tf.float32)
    # initial = tf.truncated_normal(shape, stddev=0.1)
    # return tf.Variable(initial, name=name)


def glimpse_sensor(img, norm_loc):
    # norm_loc coordinates are between -1 and 1
    loc = tf.round(((norm_loc + 1) / 2.0) * img_sz)
    loc = tf.cast(loc, tf.int32)

    img = tf.reshape(img, (-1, img_sz, img_sz, channels))

    # batch_size = img.get_shape().as_list()[0]

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
    # glimpse : [batch_size, g_depth, height, width]
    #   to conv glimpse, reshape glimpse as [batch_size * g_depth, height, width, 1]
    #    first, rehape as [batch_size * g_depth, height, width]  (x)
    #           transpose as [batch_size, height, width, g_depth] (o)
    #    second, expand dims as [batch_size*g_depth, height, width, 1]
    # rsp_glimspe = tf.reshape(glimpse, [-1, sensor_bandwidth, sensor_bandwidth])
    # ep_glimpse = tf.expand_dims(rsp_glimspe, -1)
    glimpse = tf.transpose(glimpse, [0, 2, 3, 1])

    # conv1 = conv2d(ep_glimpse, w['wg1'], b['bg1'])
    conv1 = conv2d(glimpse, w['wg1'], b['bg1'])
    conv2 = conv2d(conv1, w['wg2'], b['bg2'])
    #conv3 = conv2d(conv2, w['wg3'], b['bg3'])

    # conv3 : [batch_size * g_depth, height, width, 3]
    #   conv3 --> fc_in : [batch_size, g_depth, height, width, 3]
    #                -->  reduce_sum  [batch_size, height, width, 3]
    last_ch_size = conv2.get_shape().as_list()[-1]

    # fc_in = tf.reshape(conv2, [-1, g_depth, sensor_bandwidth, sensor_bandwidth, last_ch_size])
    # fc_in = tf.reduce_sum(fc_in, 1)
    # fc_in = tf.reshape(fc_in, [-1, sensor_bandwidth * sensor_bandwidth * last_ch_size])
    fc_in = tf.reshape(conv2, [-1, sensor_bandwidth * sensor_bandwidth * last_ch_size])

    feature = tf.nn.relu(tf.matmul(fc_in, w['wg_fc']) + b['bg_fc'])
    return feature


def glimpse_network(img, w, b, loc):
    # get input using the previous location
    glimpse_input = glimpse_sensor(img, loc)

    # the hidden units that process location & the input
    act_glimpse_hidden = get_glimpse_feature(glimpse_input, w, b)
    act_loc_hidden = tf.nn.relu(tf.matmul(loc, w['wg_lh']) + b['bg_lh'])

    # the hidden units that integrates the location & the glimpses
    glimpse_feature = tf.matmul(act_glimpse_hidden, w['wg_gh_gf'])\
                      + tf.matmul(act_loc_hidden, w['wg_lh_gf']) + b['bg_glh_gf']
    glimpse_feature = tf.nn.relu(glimpse_feature)

    return glimpse_feature


def context_network(img, w, b):
    # img_g = tf.image.resize_images(img, (patch_size, patch_size))
    # img_g = tf.reshape(img_g, shape=[-1, patch_size, patch_size, 1])

    conv1 = conv2d(img, w['wc1'], b['bc1'])
    conv2 = conv2d(conv1, w['wc2'], b['bc2'])
    pool1 = tf.nn.max_pool(conv2, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')
    #conv3 = conv2d(conv2, w['wc3'], b['bc3'])

    fc = tf.reshape(pool1, [-1, w['wc_fc'].get_shape().as_list()[0]])
    fc = tf.add(tf.matmul(fc, w['wc_fc']), b['bc_fc'])
    return tf.nn.relu(fc)


def emission_network(output, w, b, prev_loc=None):
    # the next location is computed by the location network next of core-net(Level 2 RNN Cell)
    core_net_out = tf.stop_gradient(output)

    baseline = tf.sigmoid(tf.matmul(output, w['we_bl']) + b['be_bl'])

    # compute the next location, then impose noise
    if eye_centered:
        # add the last sampled glimpse location
        # TODO max(-1, min(1, u + N(output, sigma) + prevLoc)
        if prev_loc is not None:
            mean_loc = tf.maximum(-1.0, tf.minimum(1.0, tf.matmul(core_net_out, w['we_h_nl']) + prev_loc))
        else:
            mean_loc = tf.maximum(-1.0, tf.minimum(1.0, tf.matmul(core_net_out, w['we_h_nl'])))
    else:
        if prev_loc is not None:
            mean_loc = tf.matmul(core_net_out, w['we_h_nl']) + b['be_h_nl'] + prev_loc
        else:
            mean_loc = tf.matmul(core_net_out, w['we_h_nl']) + b['be_h_nl']
        mean_loc = tf.clip_by_value(mean_loc, -1, 1)

    # add noise
    sample_loc = tf.clip_by_value(mean_loc + tf.random_normal(mean_loc.get_shape(), 0, loc_sd), -1, 1)
    # don't propagate through the locations
    sample_loc = tf.stop_gradient(sample_loc)

    return mean_loc, sample_loc, baseline


def action_network(output, w, b, step):
    if step < n_glimpse_per_element:
        # 초성
        logits = tf.add(tf.matmul(output, w['wai']), b['bai'])
    elif step < n_glimpse_per_element * 2:
        # 중성
        logits = tf.add(tf.matmul(output, w['wam']), b['bam'])
    else:
        # 종성
        logits = tf.add(tf.matmul(output, w['waf']), b['baf'])
    action = tf.nn.softmax(logits)
    return logits, action


def model(x, w, b):
    # initialize the location under uniform[-1, 1], for all example in the batch
    # batch_size = img.get_shape().as_list()[0]
    mean_locs = []
    sampled_locs = []
    outputs = []
    baselines = []
    actions = []
    action_logits = []

    img = x / 255.0

    # context feature from origin image is initial state of the top core network layer.
    context_feature = context_network(img, w, b)


    # RNN을 각 원소마다 각각 만들어보자. 초성, 중성, 종성마다

    rnn1 = tf.nn.rnn_cell.LSTMCell(lstm_size, state_is_tuple=False)
    rnn2 = tf.nn.rnn_cell.LSTMCell(lstm_size, state_is_tuple=False)

    h1 = tf.zeros([batch_size, lstm_size])
    with tf.variable_scope('rnn2', reuse=False):
        h2, state2 = rnn2(h1, context_feature)

        # initialize the location under uniform[-1, 1], for all example in the batch
        mean_loc, sampled_loc, baseline = emission_network(h2, w, b)

    mean_locs.append(mean_loc)
    sampled_locs.append(sampled_loc)
    baselines.append(baseline)

    # initialize state of 1st rnn layer as zero values
    state1 = rnn1.zero_state(batch_size, tf.float32)

    for t in range(T):
        glimpse = glimpse_network(img, w, b, sampled_loc)

        with tf.variable_scope('rnn1', reuse=(t != 0)):
            h1, state1 = rnn1(glimpse, state1)
            output = tf.sigmoid(tf.add(tf.matmul(h1, w['wo']), b['bo']))
        with tf.variable_scope('rnn2', reuse=True):
            h2, state2 = rnn2(h1, state2)
            mean_loc, sampled_loc, baseline = emission_network(h2, w, b, sampled_loc)

        logit, action = action_network(output, w, b, t)

        mean_locs.append(mean_loc)
        sampled_locs.append(sampled_loc)
        baselines.append(baseline)
        outputs.append(output)
        actions.append(action)
        action_logits.append(logit)


    elements_action = actions[n_glimpse_per_element-1::n_glimpse_per_element]
    predicted_labels = tf.stack([tf.argmax(act, -1) for act in elements_action])
    '''
    outputs : output list of 1st rnn that for decide agent's action(classification)
    mean_locs : predicted next location
    sampled_locs : random noise added location from mean_locs
    baselines : output list of 2nd rnn
    '''
    return outputs, mean_locs, sampled_locs, baselines, actions, action_logits, predicted_labels


def model2(x, w, b):
    # initialize the location under uniform[-1, 1], for all example in the batch
    # batch_size = img.get_shape().as_list()[0]
    mean_locs = []
    sampled_locs = []
    outputs = []
    baselines = []
    actions = []
    action_logits = []

    img = x / 255.0

    # context feature from origin image is initial state of the top core network layer.
    context_feature = context_network(img, w, b)


    # RNN을 각 원소마다 각각 만들어보자. 초성, 중성, 종성마다

    rnn1 = [tf.nn.rnn_cell.LSTMCell(lstm_size, state_is_tuple=False)]*n_element_per_character
    rnn2 = [tf.nn.rnn_cell.LSTMCell(lstm_size, state_is_tuple=False)]*n_element_per_character

    h1 = tf.zeros([batch_size, lstm_size])
    with tf.variable_scope('rnn2', reuse=False):
        h2, state2 = rnn2[0](h1, context_feature)

        # initialize the location under uniform[-1, 1], for all example in the batch
        mean_loc, sampled_loc, baseline = emission_network(h2, w, b)

    mean_locs.append(mean_loc)
    sampled_locs.append(sampled_loc)
    baselines.append(baseline)

    # initialize state of 1st rnn layer as zero values
    state1 = rnn1[0].zero_state(batch_size, tf.float32)

    for e in range(n_element_per_character):
        for g in range(n_glimpse_per_element):
            t = g + e * n_glimpse_per_element

            glimpse = glimpse_network(img, w, b, sampled_loc)

            with tf.variable_scope('rnn1_%d' % (e+1), reuse=(True if g else False)):
                h1, state1 = rnn1[e](glimpse, state1)
                output = tf.nn.relu(tf.add(tf.matmul(h1, w['wo']), b['bo']))
                # output = tf.sigmoid(tf.add(tf.matmul(h1, w['wo']), b['bo']))
            with tf.variable_scope('rnn2_%d' % (e+1), reuse=(False if (e > 0 and g == 0) else True)):
                h2, state2 = rnn2[e](h1, state2)
                mean_loc, sampled_loc, baseline = emission_network(h2, w, b)

            logit, action = action_network(output, w, b, t)

            mean_locs.append(mean_loc)
            sampled_locs.append(sampled_loc)
            baselines.append(baseline)
            outputs.append(output)
            actions.append(action)
            action_logits.append(logit)

    elements_action = actions[n_glimpse_per_element-1::n_glimpse_per_element]
    predicted_labels = tf.stack([tf.argmax(act, -1) for act in elements_action])
    '''
    outputs : output list of 1st rnn that for decide agent's action(classification)
    mean_locs : predicted next location
    sampled_locs : random noise added location from mean_locs
    baselines : output list of 2nd rnn
    '''
    return outputs, mean_locs, sampled_locs, baselines, actions, action_logits, predicted_labels


def losses(actions, action_logits, mean_locs, sampled_locs, baselines, labels):
    cross_entropies = []
    sq_errs = []
    logllratios = []
    equals = []

    for i in range(n_element_per_character):
        idx = (i + 1) * n_glimpse_per_element
        action = actions[idx - 1]
        logit = action_logits[idx - 1]
        pred_label = tf.argmax(action, 1)
        equal = tf.equal(pred_label, labels[:, i])

        # cross-entropy
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=labels[:, i])

        # reward : 0/1
        reward = tf.cast(equal, tf.float32)
        rewards = tf.expand_dims(reward, 1)
        rewards = tf.tile(rewards, (1, n_glimpse_per_element))
        b = baselines[idx - n_glimpse_per_element:idx]
        b = tf.stack(b, 1)
        b = tf.reshape(b, [batch_size, n_glimpse_per_element])

        # select locations at last time of each character elements
        m_locs = mean_locs[idx - n_glimpse_per_element: idx]
        s_locs = sampled_locs[idx - n_glimpse_per_element: idx]

        # log likelihood
        logll = loglikelihood(m_locs, s_locs, loc_sd)
        advs = rewards - tf.stop_gradient(b)
        logllratio = tf.reduce_mean(logll * advs)
        sq_err = tf.square(rewards - b)

        # appending to list
        cross_entropies.append(cross_entropy)
        sq_errs.append(sq_err)
        logllratios.append(logllratio)
        equals.append(equal)

    cross_entropies = tf.stack(cross_entropies)
    logllratios = tf.stack(logllratios)
    sq_errs = tf.stack(sq_errs)

    baseline_mse = tf.reduce_mean(sq_errs)
    cross_entropies = tf.reduce_mean(cross_entropies)
    logllratios = tf.reduce_mean(logllratios)

    var_list = tf.trainable_variables()
    total_loss = -logllratios + cross_entropies * 0.1 + baseline_mse  # '-' to minimize
    grads = tf.gradients(total_loss, var_list)
    max_grad_norm = 5.
    grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)

    acc_i = tf.cast(equals[0], tf.float32)
    acc_m = tf.cast(equals[1], tf.float32)
    acc_f = tf.cast(equals[2], tf.float32)

    '''
    학습할 때, zip(grads, var_list)를 optimizer.apply_gradients 하면됨.
    '''
    return logllratios, cross_entropies, baseline_mse, total_loss, grads, var_list, (acc_i, acc_m, acc_f)


def losses2(actions, action_logits, mean_locs, sampled_locs, baselines, labels):
    cross_entropies = []
    sq_errs = []
    logllratios = []
    equals = []

    for i in range(n_element_per_character):
        idx = (i + 1) * n_glimpse_per_element
        action = actions[idx - 1]
        logit = action_logits[idx - 1]
        pred_label = tf.argmax(action, 1)
        equal = tf.equal(pred_label, labels[:, i])

        # cross-entropy
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=labels[:, i])

        # reward : 0/1
        reward = tf.cast(equal, tf.float32)
        rewards = tf.expand_dims(reward, 1)
        rewards = tf.tile(rewards, (1, n_glimpse_per_element))

        b = baselines[idx - n_glimpse_per_element + 1]

        # select locations at last time of each character elements
        m_locs = mean_locs[idx - n_glimpse_per_element: idx]
        s_locs = sampled_locs[idx - n_glimpse_per_element: idx]

        # log likelihood
        logll = loglikelihood(m_locs, s_locs, loc_sd)
        advs = rewards - tf.stop_gradient(b)
        logllratio = tf.reduce_mean(logll * advs)
        sq_err = tf.square(rewards - b)

        # appending to list
        cross_entropies.append(cross_entropy)
        sq_errs.append(sq_err)
        logllratios.append(logllratio)
        equals.append(equal)

    cross_entropies = tf.stack(cross_entropies)
    logllratios = tf.stack(logllratios)
    sq_errs = tf.stack(sq_errs)

    baseline_mse = tf.reduce_mean(sq_errs)
    cross_entropies = tf.reduce_mean(cross_entropies)
    logllratios = tf.reduce_mean(logllratios)

    var_list = tf.trainable_variables()
    total_loss = -logllratios + cross_entropies * 0.1 + baseline_mse  # '-' to minimize
    grads = tf.gradients(total_loss, var_list)
    max_grad_norm = 5.
    grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)

    accs = [tf.cast(eq, tf.float32) for eq in equals]

    '''
    학습할 때, zip(grads, var_list)를 optimizer.apply_gradients 하면됨.
    '''
    return logllratios, cross_entropies, baseline_mse, total_loss, grads, var_list, accs


def pretrain_losses(x, w, b):
    c_fc_size = w['wc_fc'].get_shape().as_list()[-1]
    g_fc_size = w['wg_fc'].get_shape().as_list()[-1]

    image_size = img_len * channels

    # context net
    x_normalized = x / 255.0
    c_conv1 = conv2d(x_normalized, w['wc1'], b['bc1'])
    c_conv2 = conv2d(c_conv1, w['wc2'], b['bc2'])
    #c_conv3 = conv2d(c_conv2, w['wc3'], b['bc3'])

    c_fc = tf.reshape(c_conv2, [-1, w['wc_fc'].get_shape().as_list()[0]])
    # c_fc = tf.add(tf.matmul(c_fc, w['wc_fc']), b['bc_fc'])
    # c_fc = tf.nn.relu(c_fc)

    # glimpse net
    glimpse = []
    for k in range(batch_size):
        img = x[k, :, :, :]

        img = tf.reshape(img, (1, img.get_shape()[0].value, img.get_shape()[1].value, channels))

        # resize image to (sensorBandwidth x sensorBandwidth)
        img = tf.image.resize_bilinear(img, (sensor_bandwidth, sensor_bandwidth))
        img = tf.reshape(img, (sensor_bandwidth, sensor_bandwidth))
        glimpse.append(img)

    glimpse = tf.stack(glimpse)
    glimpse = tf.expand_dims(glimpse, -1)

    g_conv1 = conv2d(glimpse, w['wg1'], b['bg1'])
    g_conv2 = conv2d(g_conv1, w['wg2'], b['bg2'])
    #g_conv3 = conv2d(g_conv2, w['wg3'], b['bg3'])

    g_fc = tf.reshape(g_conv2, [batch_size, sensor_bandwidth * sensor_bandwidth * g_conv2.get_shape().as_list()[-1]])

    #g_fc = tf.nn.relu(tf.matmul(g_fc, w['wg_fc']) + b['bg_fc'])

    # Reconstruction
    w_c = weight_variable([w['wc_fc'].get_shape().as_list()[0], image_size], 'context_net_pretrain_weight')
    w_g = weight_variable([sensor_bandwidth * sensor_bandwidth * g_conv2.get_shape().as_list()[-1], image_size],
                          'glimpse_net_pretrain_weight')
    b_c = bias_variable([image_size], 'context_net_pretrain_bias')
    b_g = bias_variable([image_size], 'glimpse_net_pretrain_bias')

    c_reconstruction = tf.nn.sigmoid(tf.add(tf.matmul(c_fc, w_c), b_c))
    g_reconstruction = tf.nn.sigmoid(tf.add(tf.matmul(g_fc, w_g), b_g))

    # print('image_size:', image_size, 'x_normalized:', x_normalized, 'x:', x)
    x_rsp = tf.reshape(x_normalized, [-1, image_size])

    # Reconstruction Cost
    c_reconstruction_cost = tf.reduce_mean(tf.square(x_rsp - c_reconstruction))
    g_reconstruction_cost = tf.reduce_mean(tf.square(x_rsp - g_reconstruction))

    total_reconstruction_cost = tf.add(c_reconstruction_cost, g_reconstruction_cost)

    # Optimizer
    train_op_r = tf.train.AdamOptimizer(1e-3).minimize(total_reconstruction_cost)
    # train_op_r = tf.train.RMSPropOptimizer(1e-3).minimize(total_reconstruction_cost)

    return c_reconstruction, c_reconstruction_cost, g_reconstruction, g_reconstruction_cost, total_reconstruction_cost, train_op_r


def gaussian_pdf(mean, std, sample):
    Z = 1.0 / (loc_sd * tf.sqrt(2.0 * np.pi))
    a = -tf.square(sample - mean) / (2.0 * tf.square(std))
    return Z * tf.exp(a)


def loglikelihood(mean_arr, sampled_arr, sigma):
  mu        = tf.stack(mean_arr)                 # mu = [timesteps, batch_sz, loc_dim]
  sampled   = tf.stack(sampled_arr)              # same shape as mu
  logll     = gaussian_pdf(mu, sigma, sampled)
  logll     = tf.reduce_sum(logll, 2)           # sum over time steps
  logll     = tf.transpose(logll)               # [batch_sz, timesteps]

  return logll




