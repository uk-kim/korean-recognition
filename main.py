# -*-coding: utf-8-*-
from dataset.korean_utils import *
from dataset.datasets import DataSet
import unicodedata

import cv2
import numpy as np

import tensorflow as tf
from attention.attention import glimpse_sensor
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

    dataset = DataSet(args)

    images, labels = dataset.train_data.next_batch(1)
    print(images.shape, labels.shape)

    x = tf.placeholder(tf.float32, [None, 28, 28, 1])

    loc = tf.constant([[0.5, 0.5]])

    # glimpse = tf.image.extract_glimpse(x, (14, 14), [(0.5, 0.5)],
    #                                    normalized=True, centered=False)

    glimpse = glimpse_sensor(x, loc)

    '''
    tf.image.extract_glimpse
      args
        input : 이미지
        size : 패치 사이즈
        offsets : loc 좌표
        normalized : True일때, loc 좌표를 0~1로 표
        centered : 중심 좌표를 원점으로?
    '''

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


w = {
    # for context network
    'wc1': tf.get_variable('wc1', [3, 3, channels, 16], tf.float32),
    'wc2': tf.get_variable('wc2', [3, 3, 16, 64], tf.float32),
    'wc3': tf.get_variable('wc2', [1, 1, 64, 3], tf.float32),
    'wc_fc': tf.get_variable('wc_fc', [img_len * 3, lstm_size], tf.float32)
    # for emission network
    'we_bl': tf.get_variable('we_bl', [lstm_size, lstm_size], tf.float32),
    'we_h_nl': tf.get_variable('we_h_nl', [lstm_size, 2], tf.float32),
    # for action network
    'wai': tf.get_variable('wai', [lstm_size, n_initial_character], tf.float32),
    'wam': tf.get_variable('wam', [lstm_size, n_middle_character], tf.float32),
    'waf': tf.get_variable('waf', [lstm_size, n_final_character], tf.float32),
    # for glimpse network
    'wg1': tf.get_variable('wg1', [3, 3, channels, 16], tf.float32),
    'wg2': tf.get_variable('wg2', [3, 3, 16, 64], tf.float32),
    'wg3': tf.get_variable('wg2', [1, 1, 64, 3], tf.float32),
    'wg_fc': tf.get_variable('wg_fc', [sensor_bandwidth*sensor_bandwidth * 3, lstm_size], tf.float32)
    'wg_lh': tf.get_variable('wg_lh', [2, lstm_size], tf.float32),
    'wg_gh_gf': tf.get_variable('wg_gh_gf', [lstm_size, lstm_size], tf.float32),
    'wg_lh_gf': tf.get_variable('wg_lh_gf', [lstm_size, lstm_size], tf.float32),
}

b = {
    # for context network
    'bc1': tf.get_variable('bc1', [16], tf.float32),
    'bc2': tf.get_variable('bc2', [64], tf.float32),
    'bc3': tf.get_variable('bc3', [3], tf.float32),
    'bc_fc': tf.get_variable('bc_fc', [lstm_size], tf.float32),
    # for emission network
    'be_bl': tf.get_variable('be_bl', [lstm_size], tf.float32),
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
}


# import numpy as np
#
# a = [1, 2, 3, 4, 5, 6]
# a = np.array(a)
# a = a.reshape([3, 2])
# # a = np.append(a, [7, 8], 0)
#
# np.random.shuffle(a)
# c = np.array([])
# # print(a.shape, a, c, np.vstack((c, a)), np.random.random_integers(0, 10, 11))

