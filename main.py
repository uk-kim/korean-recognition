# -*-coding: utf-8-*-
from dataset.korean_utils import *
from dataset.datasets import DataSet
import unicodedata

import cv2
import numpy as np

import tensorflow as tf
from attention.attention import glimpse_sensor


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

    images, labels = dataset.train_data.next_batch(10)
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

    for i, arg in enumerate(zip(images, labels)):
        image, label = arg

        print(image.shape)
        img_flat = np.reshape(image, [1, 28, 28, 1])
        img_flat = img_flat.astype(np.float32) / 255

        g_list = sess.run(glimpse, feed_dict={x: img_flat})

        for i, patch in enumerate(g_list[0]):
            # print(i, patch.shape)
            print(i, patch.shape)
            cv2.imshow('%d th Glimpse' % (i+1), patch)

        cv2.imshow(str(label), image)
        key = cv2.waitKey(0)

        if key == ord('q'):
            break
    cv2.destroyAllWindows()


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

