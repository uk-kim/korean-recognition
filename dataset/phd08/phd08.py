#-*- coding: utf-8 -*-

import os

import numpy as np
import cv2


def convert_txt_to_images(txt_path):
    images = []
    image_tags = []

    with open(txt_path, 'r') as f:
        data_checker = 0
        for line in f:
            if not line.strip():
                # End of line
                images.append(font_array)
                data_checker = 0
            else:
                if not data_checker:
                    image_tags.append(line.strip())
                elif data_checker == 1:
                    height, width = line.strip().split(' ')
                    height = int(height)
                    width = int(width)
                    font_array = np.zeros([height, width])
                else:
                    font_array[data_checker - 2] = list(map(int, line.strip()))

                data_checker += 1
    return images, image_tags


def save_images(images, image_tags, save_path, ext='png', gaussian=True, crop=True, crop_shape=(28, 28)):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for image, tag in zip(images, image_tags):
        image = image.astype(np.float32)
        if gaussian:
            image = cv2.GaussianBlur(image, (7, 7), 0.5)
        if crop:
            image = cv2.resize(image, crop_shape)
        image *= 255
        s_path = os.path.join(save_path, '%s.%s' % (tag, ext))
        cv2.imwrite(s_path, image)
        # np.save(s_path, image)


def save_images_as_numpy(images, image_tags, save_path, ext='npy', gaussian=True, crop=True, crop_shape=(28, 28)):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for image, tag in zip(images, image_tags):
        if gaussian:
            image = image.astype(np.float32)
            image = cv2.GaussianBlur(image, (7, 7), 0.5)
        if crop:
            image = cv2.resize(image, crop_shape)
        s_path = os.path.join(save_path, '%s.%s' % (tag, ext))
        np.save(s_path, image)


def load_image(path, gaussian=True, crop=True, crop_shape=(28, 28)):
    image = None
    if os.path.exists(path):
        image = cv2.imread(path, 0)
        if gaussian:
            image = image.astype(np.float32) / 255.0
            image = cv2.GaussianBlur(image, (7, 7), 0.5)
        if crop:
            image = cv2.resize(image, crop_shape)

    return image


def load_np_image(path, gaussian=True, crop=True, crop_shape=(28, 28)):
    image = None
    if os.path.exists(path):
        image = np.load(path)
        image = image.astype(np.float32)
        if gaussian:
            image = cv2.GaussianBlur(image, (7, 7), 0.5)
        if crop:
            image = cv2.resize(image, crop_shape)
    return image


def load_images(path, ext='png', gaussian=True, crop=True, crop_shape=(28, 28)):
    images = []
    if os.path.exists(path):
        file_list = os.listdir(path)
        file_list = [file_name for file_name in file_list if file_name.split('.')[-1] == ext]
        for file_name in file_list:
            file_path = os.path.join(path, file_name)
            image = load_image(file_path, gaussian, crop, crop_shape)
            # # image = np.load(file_path)
            # image = cv2.imread(file_path, 0)
            # if gaussian:
            #     image = image.astype(np.float32) / 255.0
            #     image = cv2.GaussianBlur(image, (7, 7), 0.5)
            # if crop:
            #     image = cv2.resize(image, crop_shape)
            images.append(image)
    return images


def load_np_images(path, ext='npy', gaussian=True, crop=True, crop_shape=(28, 28)):
    images = []
    if os.path.exists(path):
        file_list = os.listdir(path)
        file_list = [file_name for file_name in file_list if file_name.split('.')[-1] == ext]
        for file_name in file_list:
            file_path = os.path.join(path, file_name)
            image = np.load(file_path)
            if gaussian:
                image = cv2.GaussianBlur(image, (7, 7), 0.5)
            if crop:
                image = cv2.resize(image, crop_shape)
            images.append(image)
    return images