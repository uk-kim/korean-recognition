'''
확인해볼 사항
 - RNN을 초성/중성/종성 각각에 대해서 만들어서 학습
 - pretraining을 수행할 때, 단계별로 학습을 수행하는 것
    1) 초성에 대한 네트워크를 학습
    2) 중성에 대한 네트워크로 Transfer learning
    3) 잘 되겠지...

'''

import tensorflow as tf
import numpy as np
import cv2
#import matplotlib.pyplot as plt

import time
import os

from dataset.datasets import DataSet
from attention.attention import model, model2, losses2, weight_variable, bias_variable, pretrain_losses
from attention.config import *


def to_image_coordinates(normalized_coordinate):
    '''
    Transform coordinate in [-1,1] to mnist
    :param coordinate_tanh: vector in [-1,1] x [-1,1]
    :return: vector in the corresponding mnist coordinate
    '''
    return np.round(((normalized_coordinate + 1) / 2.0) * img_sz)


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('param_summaries'):
        # mean = tf.reduce_mean(var)
        # tf.summary.scalar('param_mean/' + name, mean)
        # with tf.name_scope('param_stddev'):
        #     stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        # tf.summary.scalar('param_sttdev/' + name, stddev)
        # tf.summary.scalar('param_max/' + name, tf.reduce_max(var))
        # tf.summary.scalar('param_min/' + name, tf.reduce_min(var))
        tf.summary.histogram(name, var)


# def plotWholeImg(img, img_size, normalized_locs):
#     plt.imshow(np.reshape(img, [img_size, img_size]),
#                cmap=plt.get_cmap('gray'), interpolation="nearest")
#
#     plt.ylim((img_size - 1, 0))
#     plt.xlim((0, img_size - 1))
#
#     # transform the coordinate to mnist map
#     image_locs = to_image_coordinates(normalized_locs)
#     # visualize the trace of successive nGlimpses (note that x and y coordinates are "flipped")
#     plt.plot(image_locs[0, :, 1], image_locs[0, :, 0], '-o',
#              color='lawngreen')
#     plt.plot(image_locs[0, -1, 1], image_locs[0, -1, 0], 'o',
#              color='red')


def evaluate(datasets, sess, tensors):
    '''
    tensors : list  -->  [x, Y, t_acc, i_acc, m_acc, f_acc]
    '''
    data = datasets.test_data

    _acc_element = [0] * (len(tensors) - 2)

    n_iter = min(10, data.n_data // batch_size)

    for i in range(n_iter):
        images, labels = data.next_batch(batch_size)
        if images.shape[0] != batch_size:
            images, labels = dataset.train_data.next_batch(batch_size)
        accs = sess.run(tensors[2:], feed_dict={tensors[0]: images, tensors[1]: labels})
        for i in range(_acc_element):
            _acc_element[i] += accs[i]

    for i in range(len(_acc_element)):
        _acc_element[i] /= n_iter

    acc_str = " >> TOTAL ACCURACY: %.3f" % _acc_element[0]
    for i in range(len(_acc_element) - 1):
        acc_str += ", %d th ACCURACY: %.3f" % _acc_element[i+1]

    print(acc_str)
    # print(" >> TOTAL ACCURACY: %.3f, INITIAL ACCURACY: %.3f, MIDDLE ACCURACY: %.3f, FINAL ACCURACY: %.3f" %
    #       (_acc_t, _acc_i, _acc_m, _acc_f))


def pretrain(c_recon, c_recon_cost, g_recon, g_recon_cost, total_recon_cost, train_op_r, step):
    print(' Reconstruction Building.. (for pretraining)')

    print(' Pre-Training Start..!')
    start_time = time.time()
    for step in range(step):
        next_images, _ = dataset.train_data.next_batch(batch_size)
        if next_images.shape[0] != batch_size:
            next_images, _ = dataset.train_data.next_batch(batch_size)

        feed_dict = {x: next_images}

        fetches = [c_recon, c_recon_cost, g_recon, g_recon_cost, total_recon_cost, train_op_r]

        c_r, c_r_cost, g_r, g_r_cost, t_cost, op_r = sess.run(fetches, feed_dict=feed_dict)


        if step % 100 == 0:
            end_time = time.time()
            duration = end_time - start_time
            start_time = end_time
            print(
                'Step %d --> Reconstruction Pretraining: total_cost = %.5f, context_cost = %.3f, glimpse_cost = %.3f (%.3f sec)' % (
                step, t_cost, c_r_cost, g_r_cost, duration))

    print(' Pretraining Finish..!\n')


if __name__ == '__main__':
    args = {
        'dataset': 'phd08',
        'dataset_path': '/Users/kimsu/datasets/korean_image/phd08',
        'width': 28,
        'height': 28,
        'sampling': True,
        'n_sample': 50,
        'train_set_ratio': 0.9
    }
    args['data_size'] = args['width'] * args['height']

    base_path = os.path.join(os.path.curdir, '20180629')
    summary_path = os.path.join(base_path, 'summary')
    save_path = os.path.join(base_path, 'save')
    image_log_path = os.path.join(base_path, 'image_log')

    if not os.path.exists(base_path):
        os.mkdir(base_path)
    if not os.path.exists(summary_path):
        os.mkdir(summary_path)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(image_log_path):
        os.mkdir(image_log_path)

    draw = True

    # input / label image tensor
    x = tf.placeholder(tf.float32, [batch_size, 28, 28, 1], name='image')
    Y = tf.placeholder(tf.int64, shape=[batch_size, 3], name='label')

    # Weight and Bias variables
    w = {
        # for context network
        'wc1': weight_variable([3, 3, channels, 8], 'contextNet_weight_conv1'),
        'wc2': weight_variable([3, 3, 8, 3], 'contextNet_weight_conv2'),
        # 'wc3': weight_variable([3, 3, 64, 3], 'contextNet_weight_conv3'),
        'wc_fc': weight_variable([img_len // 4 * 3, lstm_size * 2], 'contextNet_weight_fc'),

        # 'wc1': tf.get_variable('contextNet_weight_conv1', [3, 3, channels, 16], tf.float32),
        # 'wc2': tf.get_variable('contextNet_weight_conv2', [3, 3, 16, 64], tf.float32),
        # 'wc3': tf.get_variable('contextNet_weight_conv3', [1, 1, 64, 3], tf.float32),
        # 'wc_fc': tf.get_variable('contextNet_weight_fc', [img_len * 3, lstm_size * 2], tf.float32),
        # for emission network
        'we_bl': weight_variable([lstm_size, 1], 'emissionNet_weight_baseline'),
        'we_h_nl': weight_variable([lstm_size, 1], 'emissionNet_weight_hidden_normalize_loc'),
        # 'we_bl': tf.get_variable('emissionNet_weight_baseline', [lstm_size, 1], tf.float32),
        # 'we_h_nl': tf.get_variable('emissionNet_weight_hidden_normalize_loc', [lstm_size, 2], tf.float32),
        # for action network
        'wai': weight_variable([lstm_size, n_initial_character], 'actionNet_weight_initial'),
        'wam': weight_variable([lstm_size, n_middle_character], 'actionNet_weight_middle'),
        'waf': weight_variable([lstm_size, n_final_character], 'actionNet_weight_final'),
        # 'wai': tf.get_variable('actionNet_weight_initial', [lstm_size, n_initial_character], tf.float32),
        # 'wam': tf.get_variable('actionNet_weight_middle', [lstm_size, n_middle_character], tf.float32),
        # 'waf': tf.get_variable('actionNet_weight_final', [lstm_size, n_final_character], tf.float32),
        # for glimpse network
        # 'wg1': weight_variable([3, 3, channels, 8], 'glimpseNet_weight_conv1'),
        'wg1': weight_variable([3, 3, g_depth, 8], 'glimpseNet_weight_conv1'),
        'wg2': weight_variable([3, 3, 8, 3], 'glimpseNet_weight_conv2'),
        # 'wg3': weight_variable([3, 3, 64, 3], 'glimpseNet_weight_conv3'),
        'wg_fc': weight_variable([sensor_bandwidth * sensor_bandwidth * 3, lstm_size], 'glimpseNet_weight_fc'),
        'wg_lh': weight_variable([2, lstm_size], 'glimpseNet_weight_loc2hidden'),
        'wg_gh_gf': weight_variable([lstm_size, lstm_size], 'glimpseNet_weight_glimpse2feature'),
        'wg_lh_gf': weight_variable([lstm_size, lstm_size], 'glimpseNet_weight_locHidden2feature'),

        # 'wg1': tf.get_variable('glimpseNet_weight_conv1', [3, 3, channels, 16], tf.float32),
        # 'wg2': tf.get_variable('glimpseNet_weight_conv2', [3, 3, 16, 64], tf.float32),
        # 'wg3': tf.get_variable('glimpseNet_weight_conv3', [1, 1, 64, 3], tf.float32),
        # 'wg_fc': tf.get_variable('glimpseNet_weight_fc', [sensor_bandwidth * sensor_bandwidth * 3, lstm_size], tf.float32),
        # 'wg_lh': tf.get_variable('glimpseNet_weight_loc2hidden', [2, lstm_size], tf.float32),
        # 'wg_gh_gf': tf.get_variable('glimpseNet_weight_glimpse2feature', [lstm_size, lstm_size], tf.float32),
        # 'wg_lh_gf': tf.get_variable('glimpseNet_weight_locHidden2feature', [lstm_size, lstm_size], tf.float32),
        # for core network
        'wo': weight_variable([lstm_size, lstm_size], 'coreNet_weight_out'),
        # 'wo': tf.get_variable('coreNet_weight_out', [lstm_size, lstm_size], tf.float32)
    }
    b = {
        # for context network
        'bc1': bias_variable([8], 'contextNet_bias_conv1'),
        'bc2': bias_variable([3], 'contextNet_bias_conv2'),
        # 'bc3': bias_variable([3], 'contextNet_bias_conv3'),
        'bc_fc': bias_variable([lstm_size * 2], 'contextNet_bias_fc'),

        # 'bc1': tf.get_variable('contextNet_bias_conv1', [16], tf.float32),
        # 'bc2': tf.get_variable('contextNet_bias_conv2', [64], tf.float32),
        # 'bc3': tf.get_variable('contextNet_bias_conv3', [3], tf.float32),
        # 'bc_fc': tf.get_variable('contextNet_bias_fc', [lstm_size * 2], tf.float32),
        # for emission network
        'be_bl': bias_variable([1], 'emissionNet_bias_baseline'),
        'be_h_nl': bias_variable([2], 'emissionNet_bias_hidden_normalize_loc'),
        # 'be_bl': tf.get_variable('emissionNet_bias_baseline', [1], tf.float32),
        # 'be_h_nl': tf.get_variable('emissionNet_bias_hidden_normalize_loc', [2], tf.float32),
        # for action network
        'bai': bias_variable([n_initial_character], 'actionNet_bias_initial'),
        'bam': bias_variable([n_middle_character], 'actionNet_bias_middle'),
        'baf': bias_variable([n_final_character], 'actionNet_bias_final'),
        # 'bai': tf.get_variable('actionNet_bias_initial', [n_initial_character], tf.float32),
        # 'bam': tf.get_variable('actionNet_bias_middle', [n_middle_character], tf.float32),
        # 'baf': tf.get_variable('actionNet_bias_final', [n_final_character], tf.float32),
        # for glimpse network
        'bg1': bias_variable([8], 'glimpseNet_bias_conv1'),
        'bg2': bias_variable([3], 'glimpseNet_bias_conv2'),
        # 'bg3': bias_variable([3], 'glimpseNet_bias_conv3'),
        'bg_fc': bias_variable([lstm_size], 'glimpseNet_bias_fc'),
        'bg_lh': bias_variable([lstm_size], 'glimpseNet_bias_loc2hidden'),
        'bg_glh_gf': bias_variable([lstm_size], 'glimpseNet_bias_feature'),

        # 'bg1': tf.get_variable('glimpseNet_bias_conv1', [16], tf.float32),
        # 'bg2': tf.get_variable('glimpseNet_bias_conv2', [64], tf.float32),
        # 'bg3': tf.get_variable('glimpseNet_bias_conv3', [3], tf.float32),
        # 'bg_fc': tf.get_variable('glimpseNet_bias_fc', [lstm_size], tf.float32),
        # 'bg_lh': tf.get_variable('glimpseNet_bias_loc2hidden', [lstm_size], tf.float32),
        # 'bg_glh_gf': tf.get_variable('glimpseNet_bias_feature', [lstm_size], tf.float32),
        # for core network
        'bo': bias_variable([lstm_size], 'coreNet_bias_out'),
        # 'bo': tf.get_variable('coreNet_bias_out', [lstm_size], tf.float32)
    }

    # Model Build
    print(' Building model..')
    outputs, mean_locs, sampled_locs, baselines, actions, action_logits, predicted_labels = model2(x, w, b)

    # Loss Fuction
    print(' Defining loss function..')
    logllratios, cross_entropies, baseline_mse, total_loss, grads, var_list, accs = losses2(
        actions=actions,
        action_logits=action_logits,
        mean_locs=mean_locs,
        sampled_locs=sampled_locs,
        baselines=baselines,
        labels=Y)

    acc_element = [tf.reduce_mean(acc) for acc in accs]
    acc_total = tf.reduce_mean(tf.stack(acc_element))

    # Reconstruction Building.. (for pretraining)
    if pretrain_flag:
        c_recon, c_recon_cost, g_recon, g_recon_cost, total_recon_cost, train_op_r = pretrain_losses(x, w, b)

    # Optimizer
    print(' Creating optimizer..')
    optimizer = tf.train.AdamOptimizer(lr)
    # train_op = optimizer.apply_gradients(zip(grads, var_list))
    train_op = optimizer.minimize(total_loss)

    # tensorboard visualization for the parameters
    print(' Summarizing tensor variables and scalar to visualize by tensorboard..')
    # for context network
    variable_summaries(w['wc1'], "contextNet_weight_conv1")
    variable_summaries(b['bc1'], "contextNet_bias_conv1")
    variable_summaries(w['wc2'], "contextNet_weight_conv2")
    variable_summaries(b['bc2'], "contextNet_bias_conv2")
    # variable_summaries(w['wc3'], "contextNet_weight_conv3")
    # variable_summaries(b['bc3'], "contextNet_bias_conv3")
    # for emission network
    variable_summaries(w['we_bl'], "emissionNet_weight_baseline")
    variable_summaries(b['be_bl'], "emissionNet_bias_baseline")
    variable_summaries(w['we_h_nl'], "emissionNet_weight_hidden_normalize_loc")
    variable_summaries(b['be_h_nl'], "emissionNet_bias_hidden_normalize_loc")
    # for action network
    variable_summaries(w['wai'], "actionNet_weight_initial")
    variable_summaries(b['bai'], "actionNet_bias_initial")
    variable_summaries(w['wam'], "actionNet_weight_middle")
    variable_summaries(b['bam'], "actionNet_bias_middle")
    variable_summaries(w['waf'], "actionNet_weight_final")
    variable_summaries(b['baf'], "actionNet_bias_final")
    # for glimpse network
    variable_summaries(w['wg1'], "glimpseNet_weight_conv1")
    variable_summaries(b['bg1'], "glimpseNet_bias_conv1")
    variable_summaries(w['wg2'], "glimpseNet_weight_conv2")
    variable_summaries(b['bg2'], "glimpseNet_bias_conv2")
    # variable_summaries(w['wg3'], "glimpseNet_weight_conv3")
    # variable_summaries(b['bg3'], "glimpseNet_bias_conv3")
    variable_summaries(w['wg_fc'], "glimpseNet_weight_fc")
    variable_summaries(b['bg_fc'], "glimpseNet_bias_fc")

    variable_summaries(w['wg_lh'], "glimpseNet_weight_loc2hidden")
    variable_summaries(b['bg_lh'], "glimpseNet_bias_loc2hidden")
    variable_summaries(w['waf'], "actionNet_weight_final")
    variable_summaries(b['baf'], "actionNet_bias_final")

    variable_summaries(w['wg_gh_gf'], "glimpseNet_weight_glimpse2feature")
    variable_summaries(w['wg_lh_gf'], "glimpseNet_weight_locHidden2feature")
    variable_summaries(b['bg_glh_gf'], "glimpseNet_bias_feature")
    # for core network
    variable_summaries(w['wo'], "coreNet_weight_out")
    variable_summaries(b['bo'], "coreNet_bias_out")

    # tensorboard visualization for the performance metrics
    tf.summary.scalar("Loss/loglikelihood_ratio", logllratios)
    tf.summary.scalar("Loss/baseline_mse", baseline_mse)
    tf.summary.scalar("Loss/cross_entropies", cross_entropies)
    for i, acc_arg in enumerate(acc_element):
        tf.summary.scalar("Accuracy/accuracy_%d_element" % (i+1), acc_arg)
    # tf.summary.scalar("Accuracy/accuracy_initial", acc_i)
    # tf.summary.scalar("Accuracy/accuracy_middle", acc_m)
    # tf.summary.scalar("Accuracy/accuracy_final", acc_f)
    tf.summary.scalar("Accuracy/accuracy_total", acc_total)
    tf.summary.scalar("Loss/total_loss", total_loss)
    summary_op = tf.summary.merge_all()

    ############################################################### Training #####################
    # Model Training

    # Data set : PHD08
    print(' Loading datasets..')
    dataset = DataSet(args)

    # Session initialize
    print(' Initializing session..')
    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    saver = tf.train.Saver()

    init = tf.global_variables_initializer()
    sess.run(init)

    summary_writer = tf.summary.FileWriter(summary_path, graph=sess.graph)

    # Pre-Training context, glimpse net using reconstruction error
    if pretrain_flag:
        pretrain(c_recon, c_recon_cost, g_recon, g_recon_cost, total_recon_cost, train_op_r, pretrain_step)


    print(' Training Start..!')
    for step in range(total_step):
        start_time = time.time()
        next_images, next_labels = dataset.train_data.next_batch(batch_size)
        if next_images.shape[0] != batch_size:
            next_images, next_labels = dataset.train_data.next_batch(batch_size)

        feed_dict = {x: next_images, Y: next_labels}

        fetches = [train_op, total_loss, baseline_mse, cross_entropies, acc_element, acc_total,
                   mean_locs, predicted_labels]

        _, t_loss, b_mse, x_ent, acc_list, t_acc, m_locs, p_labels = sess.run(fetches, feed_dict=feed_dict)

        duration = time.time() - start_time
        if step == 0:
            print(type(acc_list), acc_list)
        if step % 50 == 0:
            print('Step %d: total_loss = %.5f, t_acc = %.3f (%.3f sec) mse = %.5f, x_ent = %.5f' % (step, t_loss,
                                                                                                    t_acc, duration,
                                                                                                    b_mse, x_ent*0.1))

            summary_str = sess.run(summary_op, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, step)

            if step > 0 and step % 10000 == 0:
                saver.save(sess, os.path.join(save_path, str(step) + ".ckpt"))
                evaluate(dataset, sess, [x, Y, acc_total] + [arg for arg in acc_list])
                mls = np.array(m_locs)

                print('   mean locations: ', mls[:,0,:])
            if step % 500 == 0 and n_element_per_character == 3:
                for i in range(min(batch_size, 3)):
                    predicted_label = p_labels[:, i]
                    true_label = next_labels[i]

                    label_str = "   >>> True Label : [" \
                                + dataset.i2c_i[true_label[0]] \
                                + dataset.i2c_m[true_label[1]] \
                                + dataset.i2c_f[true_label[2]] \
                                + "]   Predicted Label: [" \
                                + dataset.i2c_i[predicted_label[0]] \
                                + dataset.i2c_m[predicted_label[1]] \
                                + dataset.i2c_f[predicted_label[2]] + "]"
                    print(label_str)

            if step % 1000 == 0 and draw:
                def visualize_glimpse_movement(image, locs):
                    r_image = cv2.resize(image, (image.shape[1] * 15, image.shape[0] * 15))
                    r_image = np.expand_dims(r_image, -1)

                    rows = r_image.shape[0]
                    cols = r_image.shape[1]
                    n_channel = r_image.shape[2]
                    disp = r_image.copy()
                    if n_channel == 1:
                        disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)

                    pts = []
                    for loc in locs:
                        x = int((loc[0] + 1) * 0.5 * cols + 0.5)
                        y = int((loc[1] + 1) * 0.5 * rows + 0.5)
                        pts.append((x, y))

                        cv2.circle(disp, (x, y), 1, (0, 255, 0), 3)

                    start_color = 100
                    color_gap = (255 - start_color) // (len(pts) - 1)
                    for i in range(len(pts) - 1):
                        color = min(255, start_color + i * color_gap)
                        cv2.line(disp, pts[i], pts[i+1], (0, color, 0), 2)
                        cv2.line(disp, pts[i], pts[i + 1], (0, 255, 0), 2)
                        cv2.circle(disp, pts[i], 4, (0, 255, 0), 3)

                    cv2.circle(disp, pts[0], 4, (255, 0, 0), 4)
                    cv2.circle(disp, pts[-1], 4, (0, 0, 255), 4)
                    return disp

                m_locs = np.array(m_locs)
                m_locs = np.transpose(m_locs, [1, 0, 2])
                # print(mean_locs)
                # print('m_locs:', m_locs.shape, m_locs)
                disp_list = []
                for i in range(min(batch_size, 3)):
                    disp = visualize_glimpse_movement(next_images[i], m_locs[i])
                    cv2.imwrite(os.path.join(image_log_path, 'glimpse_%d_(%d).png' % (step, i)), disp)

    sess.close()
    print(' Training has been finished..')

