# -*-coding: utf-8-*-
from dataset.korean_utils import *
from dataset.datasets import DataSet
import unicodedata

import cv2
import numpy as np


if __name__ == '__main__':
    args = {
        'dataset': 'phd08',
        'dataset_path': '/Users/kimsu/datasets/korean_image/phd08',
        'width': 28,
        'height': 28,
        'sampling': True,
        'n_sample': 50
    }
    args['data_size'] = args['width'] * args['height']

    dataset = DataSet(args)

import numpy as np

a = [1, 2, 3, 4, 5, 6]
a = np.array(a)
a = a.reshape([3, 2])
# a = np.append(a, [7, 8], 0)

np.random.shuffle(a)
c = np.array([])
# print(a.shape, a, c, np.vstack((c, a)), np.random.random_integers(0, 10, 11))