import unicodedata
import os

import numpy as np
from tqdm import tqdm

from .phd08 import phd08
from dataset import korean_utils as utils


def get_letters(path):
    if not os.path.exists(path):
        return []
    else:
        letters = []
        with open(path, 'r') as f:
            for line in f:
                letter = line.strip()
                letters.append(unicodedata.normalize('NFC', letter))
        return letters


class DataSet:
    def __init__(self, args):
        # self.dataset is interface for usage of various korean data sets.
        if args['dataset'] == 'phd08':
            self.dataset = phd08
        else:
            # If you has another data sets, append 'elif' term likes above that.
            self.dataset = phd08

        self.main_path = args['dataset_path']
        self.IMAGE_HEIGHT = args['height']
        self.IMAGE_WIDTH = args['width']

        # Load Korean Characters
        #   'i' is first location of korean letter that mean as Cho Sung
        #   'm' is second location of korean letter that mean as Jung Sung
        #   'f' is final location of korean letter that mean as Jong Sung
        # i_list, m_list, f_list = utils.load_korean_chars(os.path.join(self.main_path, 'korean_character_table.txt'))
        i_list, m_list, f_list = utils.load_korean_chars(None)

        self.c2i_i = {c: i for i, c in enumerate(i_list)}
        self.c2i_m = {c: i for i, c in enumerate(m_list)}
        self.c2i_f = {c: i for i, c in enumerate(f_list)}

        self.i2c_i = {i: c for i, c in enumerate(i_list)}
        self.i2c_m = {i: c for i, c in enumerate(m_list)}
        self.i2c_f = {i: c for i, c in enumerate(f_list)}

        self.idx_to_label = get_letters(os.path.join(self.main_path, 'korean_letters.txt'))
        self.label_to_idx = {self.idx_to_label[i]: i for i in range(len(self.idx_to_label))}

        # image_list, label_list = self.load_images()
        path_list, label_list = self.get_image_paths(base_path=self.main_path,
                                                     ext='png',
                                                     sampling=args['sampling'],
                                                     n_sample=args['n_sample'])

    def get_image_paths(self, base_path, ext='png', sampling=False, n_sample=100):
        image_path_list = None
        label_list = None
        for letter in tqdm(self.idx_to_label):
            _, label = utils.decompose_korean_letter(letter,
                                                     self.i2c_i,
                                                     self.i2c_m,
                                                     self.i2c_f,
                                                     self.c2i_i,
                                                     self.c2i_m,
                                                     self.c2i_f)
            path = os.path.join(base_path, ext + '/' + letter)
            file_list = os.listdir(path)
            file_list = [os.path.join(letter, file_name) for file_name in file_list if file_name.split('.')[-1] == ext]

            file_list = np.array(file_list)
            labels = label * len(file_list)
            labels = np.array(labels).reshape([-1, 3])

            if sampling:
                idxs = np.arange(len(file_list))
                np.random.shuffle(idxs)
                idxs = idxs[:n_sample]

                file_list = file_list[idxs]
                labels = labels[idxs]

            if image_path_list is None:
                image_path_list = file_list
                label_list = labels
            else:
                image_path_list = np.concatenate((image_path_list, file_list))
                label_list = np.vstack((label_list, labels))

        return image_path_list, label_list

    def load_images(self, ext='png'):
        image_list = []
        label_list = []

        base_path = os.path.join(self.main_path, ext)
        for letter in self.idx_to_label[1:5]:
            _, label = utils.decompose_korean_letter(letter,
                                                     self.i2c_i,
                                                     self.i2c_m,
                                                     self.i2c_f,
                                                     self.c2i_i,
                                                     self.c2i_m,
                                                     self.c2i_f)

            path = os.path.join(base_path, letter)
            images = self.dataset.load_images(path, ext, False, False)

            image_list += images
            label_list += label * len(images)

        image_list = np.array(image_list)
        label_list = np.array(label_list)
        label_list = label_list.reshape([-1, 3])
        return image_list, label_list

