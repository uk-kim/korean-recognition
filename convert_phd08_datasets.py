'''
This script is for converting phd08 dataset which is consisted of text files to image file format.
First, load and parse text files and create new folder as korean letter that corresponding to text file.
Second, Convert text file to image format with some steps include Cropping, Gaussian blur..
Final, Save the converted images to folder before you create.
'''

import os
from dataset.phd08 import phd08
from tqdm import tqdm

main_path = '/Users/kimsu/datasets/korean_image/phd08'


def get_letters(path):
    if not os.path.exists(path):
        return []
    else:
        letters = []
        with open(path, 'r') as f:
            for line in f:
                letters.append(line.strip())
        return letters


letters = get_letters(os.path.join(main_path, 'korean_letters.txt'))

txt_path = os.path.join(main_path, 'txt')
img_path = os.path.join(main_path, 'png')

for letter in tqdm(letters):
    data_path = os.path.join(txt_path, '%s.txt' % letter)

    images, tags = phd08.convert_txt_to_images(data_path)
    phd08.save_images(images, tags, os.path.join(img_path, letter), 'png', True, True, (28, 28))
