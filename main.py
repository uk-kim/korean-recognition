# -*-coding: utf-8-*-
from dataset.korean_utils import *
from dataset.datasets import DataSet
import unicodedata

import cv2
import numpy as np

import tensorflow as tf
from attention.attention import glimpse_sensor, model
from attention.config import *


if __name__ == '__main__':
    args = {
        'dataset': 'phd08',
        'dataset_path': '/Users/kimsu/datasets/korean_image/phd08',
        'width': 28,
        'height': 28,
        'sampling': True,
        'n_sample': 50,
        'train_set_ratio': 0.7
    }
    args['data_size'] = args['width'] * args['height']

    # Weight and Bias variables
    w = {
        # for context network
        'wc1': tf.get_variable('wc1', [3, 3, channels, 16], tf.float32),
        'wc2': tf.get_variable('wc2', [3, 3, 16, 64], tf.float32),
        'wc3': tf.get_variable('wc3', [1, 1, 64, 3], tf.float32),
        'wc_fc': tf.get_variable('wc_fc', [img_len * 3, lstm_size * 2], tf.float32),
        # for emission network
        'we_bl': tf.get_variable('we_bl', [lstm_size, 1], tf.float32),
        'we_h_nl': tf.get_variable('we_h_nl', [lstm_size, 2], tf.float32),
        # for action network
        'wai': tf.get_variable('wai', [lstm_size, n_initial_character], tf.float32),
        'wam': tf.get_variable('wam', [lstm_size, n_middle_character], tf.float32),
        'waf': tf.get_variable('waf', [lstm_size, n_final_character], tf.float32),
        # for glimpse network
        'wg1': tf.get_variable('wg1', [3, 3, channels, 16], tf.float32),
        'wg2': tf.get_variable('wg2', [3, 3, 16, 64], tf.float32),
        'wg3': tf.get_variable('wg3', [1, 1, 64, 3], tf.float32),
        'wg_fc': tf.get_variable('wg_fc', [sensor_bandwidth * sensor_bandwidth * 3, lstm_size], tf.float32),
        'wg_lh': tf.get_variable('wg_lh', [2, lstm_size], tf.float32),
        'wg_gh_gf': tf.get_variable('wg_gh_gf', [lstm_size, lstm_size], tf.float32),
        'wg_lh_gf': tf.get_variable('wg_lh_gf', [lstm_size, lstm_size], tf.float32),
        # for core network
        'wo': tf.get_variable('wo', [lstm_size, lstm_size], tf.float32)
    }
    b = {
        # for context network
        'bc1': tf.get_variable('bc1', [16], tf.float32),
        'bc2': tf.get_variable('bc2', [64], tf.float32),
        'bc3': tf.get_variable('bc3', [3], tf.float32),
        'bc_fc': tf.get_variable('bc_fc', [lstm_size * 2], tf.float32),
        # for emission network
        'be_bl': tf.get_variable('be_bl', [1], tf.float32),
        'be_h_nl': tf.get_variable('be_h_nl', [2], tf.float32),
        # for action network
        'bai': tf.get_variable('bai', [n_initial_character], tf.float32),
        'bam': tf.get_variable('bam', [n_middle_character], tf.float32),
        'baf': tf.get_variable('baf', [n_final_character], tf.float32),
        # for glimpse network
        'bg1': tf.get_variable('bg1', [16], tf.float32),
        'bg2': tf.get_variable('bg2', [64], tf.float32),
        'bg3': tf.get_variable('bg3', [3], tf.float32),
        'bg_fc': tf.get_variable('bg_fc', [lstm_size], tf.float32),
        'bg_lh': tf.get_variable('bg_lh', [lstm_size], tf.float32),
        'bg_glh_gf': tf.get_variable('bg_glh_gf', [lstm_size], tf.float32),
        # for core network
        'bo': tf.get_variable('bo', [lstm_size], tf.float32)
    }
    x = tf.placeholder(tf.float32, [batch_size, 28, 28, 1])

    # Data set : PHD08
    dataset = DataSet(args)

    images, labels = dataset.train_data.next_batch(1)



    # Glimpse Sensor Test
    """
    loc = tf.constant([[0.5, 0.5]])
    glimpse = glimpse_sensor(x, loc)
    
    # Create Session
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    
    for i, arg in enumerate(zip(images, labels)):
        image, label = arg

        print(image.shape)
        img_flat = np.reshape(image, [1, 28, 28, 1])
        img_flat = img_flat.astype(np.float32) / 255

        g_list = sess.run(glimpse, feed_dict={x: img_flat})
        print(g_list.shape)
        for i, patch in enumerate(g_list[0]):
            # print(i, patch.shape)
            print(i, patch.shape)
            cv2.imshow('%d th Glimpse' % (i+1), patch)

        cv2.imshow(str(label), image)
        key = cv2.waitKey(0)

        if key == ord('q'):
            break
    cv2.destroyAllWindows()
    """

    # Build Model
    outputs, mean_locs, sampled_locs, baselines, actions = model(x, w, b)


    def visualize_glimpse_movement(img, locs):
        rows = img.shape[0]
        cols = img.shape[1]
        n_channel = img.shape[2]
        disp = img.copy()
        if n_channel == 1:
            disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)

        pts = []
        for loc in locs:
            x = int((loc[0, 0] + 1) * 0.5 * cols + 0.5)
            y = int((loc[0, 1] + 1) * 0.5 * rows + 0.5)
            pts.append((x, y))

            cv2.circle(disp, (x, y), 1, (0, 255, 0), 2)
        cv2.circle(disp, pts[0], 1, (255, 0, 0), 2)
        cv2.circle(disp, pts[-1], 1, (0, 0, 255), 2)

        for i in range(len(pts) - 1):
            cv2.line(disp, pts[i], pts[i + 1], (0, 255, 0), 1)
        return disp


    image = images[0]
    image = np.reshape(image, [1, img_sz, img_sz, 1])
    image = image.astype(np.float32) / 255

    # Predict Glimpse Locations
    # Create Session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    locs = sess.run(mean_locs, feed_dict={x: image})

    r_image = cv2.resize(image[0], (image.shape[2] * 15, image.shape[1] * 15))
    r_image = np.expand_dims(r_image, -1)
    disp = visualize_glimpse_movement(r_image, locs)

    cv2.imshow("disp", disp)
    cv2.waitKey()

    cv2.destroyAllWindows()

    sess.close()
