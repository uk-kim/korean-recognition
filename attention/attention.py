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
    #    first, rehape as [batch_size * g_depth, height, width]
    #    second, expand dims as [batch_size*g_depth, height, width, 1]
    rsp_glimspe = tf.reshape(glimpse, [-1, sensor_bandwidth, sensor_bandwidth])
    ep_glimpse = tf.expand_dims(rsp_glimspe, -1)

    conv1 = conv2d(ep_glimpse, w['wg1'], b['bg1'])
    conv2 = conv2d(conv1, w['wg2'], b['bg2'])
    conv3 = conv2d(conv2, w['wg3'], b['bg3'])

    # conv3 : [batch_size * g_depth, height, width, 3]
    #   conv3 --> fc_in : [batch_size, g_depth, height, width, 3]
    #                -->  reduce_sum  [batch_size, height, width, 3]
    last_ch_size = conv3.get_shape().as_list()[-1]

    fc_in = tf.reshape(conv3, [-1, g_depth, sensor_bandwidth, sensor_bandwidth, last_ch_size])
    fc_in = tf.reduce_sum(fc_in, 1)
    fc_in = tf.reshape(fc_in, [-1, sensor_bandwidth * sensor_bandwidth * last_ch_size])

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
    conv3 = conv2d(conv2, w['wc3'], b['bc3'])

    fc = tf.reshape(conv3, [-1, w['wc_fc'].get_shape().as_list()[0]])
    fc = tf.add(tf.matmul(fc, w['wc_fc']), b['bc_fc'])
    return tf.nn.relu(fc)


def emission_network(output, w, b):
    # the next location is computed by the location network next of core-net(Level 2 RNN Cell)
    core_net_out = tf.stop_gradient(output)

    baseline = tf.sigmoid(tf.matmul(output, w['we_bl']) + b['be_bl'])

    # compute the next location, then impose noise
    if eye_centered:
        # add the last sampled glimpse location
        # TODO max(-1, min(1, u + N(output, sigma) + prevLoc)
        mean_loc = tf.maximum(-1.0, tf.minimum(1.0, tf.matmul(core_net_out, w['we_h_nl'])))
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
        action = tf.add(tf.matmul(output, w['wai']), b['bai'])
    elif step < n_glimpse_per_element * 2:
        # 중성
        action = tf.add(tf.matmul(output, w['wam']), b['bam'])
    else:
        # 종성
        action = tf.add(tf.matmul(output, w['waf']), b['baf'])
    action = tf.nn.softmax(action)
    return action


def model(img, w, b):
    # initialize the location under uniform[-1, 1], for all example in the batch
    # batch_size = img.get_shape().as_list()[0]
    mean_locs = []
    sampled_locs = []
    outputs = []
    baselines = []
    actions = []

    # context feature from origin image is initial state of the top core network layer.
    context_feature = context_network(img, w, b)

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
            mean_loc, sampled_loc, baseline = emission_network(h2, w, b)

        action = action_network(output, w, b, t)

        mean_locs.append(mean_loc)
        sampled_locs.append(sampled_loc)
        baselines.append(baseline)
        outputs.append(output)
        actions.append(action)


    '''
    outputs : output list of 1st rnn that for decide agent's action(classification)
    mean_locs : predicted next location
    sampled_locs : random noise added location from mean_locs
    baselines : output list of 2nd rnn
    '''
    return outputs, mean_locs, sampled_locs, baselines, actions


def losses(actions, mean_locs, sampled_locs, baselines, labels):
    1


def calc_reward(actions, mean_locs, sampled_locs, baselines, labels):
    '''
    reward를 계산할 시, 초성 중성 종성에 따라 차원의 수에 유의해야 함
    그리고 seq2seq 모델의 형태와 같이 시작과 끝을 나타내는 시그널(?)을 만드는 장치를 추가 하면 좋을듯
     --> 가: ㄱ + ㅏ  (초성 + 중성)
         감: ㄱ + ㅏ + ㅁ (초성 + 중성 + 종성)

         즉, 중성 다음에 종성이 올지 말지에 대해서는 중성 다음에 시퀀스 종료 시그널을 통해 파악하면 될것 같다.
         유념해서 반영하자.
    '''
    '''
    labels : [batchsize, 3] 형태로 구성
            batch 크기만큼에 대해서 각 초/중/종성에 해당하는 타겟의 index로 이루어짐
    '''

    '''
    reference : https://github.com/PrincipalComponent/ram/blob/master/DRAM.py

    reward
    baseline mse : baseline과 reward를 가지고 mse 계산
    action에 대한 cross-entropy
    location에 대한 loglikelihood : 위 reference 의 src/utils.py 참고

    loss = baseline mse + cross-entropy + loglikelihood
    '''
    # 각 성분별 마지막으로 glimpse한 시점에서의 출력 결과
    #   actions : [n_glimpse_per_element * n_element_per_character, batch_size, 각 타입의 수]

    # Losses/reward
    # 일반적인 DRAM/Multiple Object Recognition 방법에서 loss fucntion을 아래와 같이 coding함.
    # 현재는 shape, dimension이 하나도 고려가 안된 상태이니까 맞추어야 함.
    # 그리고, 내 문제에서는 초/중/종성에 따라 나누어 디멘전이 다 다르기 때문에 고려해주어야함.

    # cross-entropy
    #   labels : [batch_size, n_element_per_character]

    cross_entropies = []
    equals = []
    sq_errs = []
    for i in range(n_element_per_character):
        idx = (i+1) * n_glimpse_per_element - 1
        logits = actions[idx]  # logits --> [batch_size, i번째 요소의 가짓 수]

        # pred_label, equal : [batch_size]
        pred_label = tf.argmax(logits, 1)
        equal = tf.equal(pred_label, labels[:, i])
        equals.append(equal)

        # cross_entropy : [batch_size]
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels[:, i])
        cross_entropy = tf.reduce_mean(cross_entropy)
        cross_entropies.append(cross_entropy)

        # REINFORCE: 0/1 reward
        reward = tf.cast(equal, tf.float32)  # reward : [batch_size]
        rewards = tf.expand_dims(reward, 1)  # shape: [batch_size, 1]
        rewards = tf.tile(rewards, (1, n_glimpse_per_element))  # shape: [batch_size, n_glimpse_per_element]

        # log-likelihood
        logll = loglikelihood(mean_locs[idx+1], sampled_locs[idx+1], loc_sd)
        advs = rewards - tf.stop_gradient(baselines[idx+1])
        logllratio = tf.reduce_mean(logll * advs)

        reward = tf.reduce_mean(reward)
        sq_errs.append(tf.square(rewards - baselines[idx+1]))

    sq_errs_tensor = tf.stack(sq_errs)
    baselines_mse = tf.reduce_mean(sq_errs_tensor)

    var_list = tf.trainable_variables()

    equals = tf.stack(equals)  # equals : [n_element_per_character, batch_size]
    cross_entropies = tf.stack(cross_entropies)  # cross_entropies : [n_element_per_character, batch_size]

    equals = tf.transpose(equals, [1, 0])
    cross_entropies = tf.transpose(cross_entropies, [1, 0])
    accuracy = tf.reduce_mean(tf.cast(equals, tf.float32))
    # accuracy = tf.reduce_mean(equals, axis=0)

    return var_list,

    # REINFORCE: 0/1 reward



    # xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=_actions, labels=labels)
    # xent = tf.reduce_mean(xent)
    # pred_labels = tf.argmax(_actions, 1)
    # equal = tf.equal(pred_labels, labels)
    # accuracy = tf.reduce_mean(tf.cast(equal, tf.float32))
    #
    # # REINFORCE: 0/1 reward
    # reward = tf.cast(tf.equal(pred_labels, labels), tf.flaot32)
    # rewards = tf.expand_dims(reward, 1) # [batch_size, 1]
    # rewards = tf.tile(rewards, (1, n_glimpse_per_element))  # [batch_size, timesteps]
    # logll = loglikelihood(mean_locs, sampled_locs, loc_sd)
    # advs = rewards - tf.stop_gradient(baselines)
    # logllratio = tf.reduce_mean(logll * advs)
    # reward = tf.reduce_mean(reward)
    #
    # baselines_mse = tf.reduce_mean(tf.square((rewards - baselines)))
    # var_list = tf.trainable_variables()
    #
    # # total loss
    # total = -logllratio + xent + baselines_mse  # '-' to minimize
    # grads = tf.gradients(total, var_list)
    # max_grad_norm = 5.  # source code 참조
    # grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)

    '''
    학습할 때, zip(grads, var_list)를 optimizer.apply_gradients 하면됨.
    '''


def gaussian_pdf(mean, std, sample):
    Z = 1.0 / (loc_sd * tf.sqrt(2.0 * np.pi))
    a = -tf.square(sample - mean) / (2.0 * tf.square(std))
    return Z * tf.exp(a)


def loglikelihood(mean_arr, sampled_arr, sigma):
  mu        = tf.stack(mean_arr)                 # mu = [timesteps, batch_sz, loc_dim]
  sampled   = tf.stack(sampled_arr)              # same shape as mu
  logll     = gaussian_pdf(mu, sigma, sampled)
  # gaussian  = tf.contrib.distributions.Normal(mu, sigma)
  # print(gaussian)
  # logll     = gaussian.log_pdf(sampled)         # [timesteps, batch_sz, loc_dim]
  print(logll)
  logll     = tf.reduce_sum(logll, 2)           # sum over time steps
  logll     = tf.transpose(logll)               # [batch_sz, timesteps]

  return logll




