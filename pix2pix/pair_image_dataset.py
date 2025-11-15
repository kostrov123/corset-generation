import os
import pickle

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from random import randint

import pix2pix.config as config


class PairImageDataset(Dataset):
    """ Датасет для работы с парой изображений  человек-корсет.
    """

    def __init__(self, root_dir, is_train):
        """
        :param root_dir: Директория с исходными картинками.
        :param is_train: Обучение/валидация.
        """
        self.root_dir = root_dir
        self.is_train = is_train
        self.list_files_src = []
        self.list_files_tar = []
        self.list_angles = []

        # Загрузка словаря иображение - эмбеддинг углов позвоночника.
        with open('embs.pkl', 'rb') as f:
            self.rg_embeddings = pickle.load(f)

        for filename in tqdm(os.listdir(root_dir)):
            if filename.split('_')[0][-1] != '2' and 'human' in filename:
                corset_path = os.path.join(root_dir, filename.replace("human", "corset"))
                if os.path.exists(corset_path):
                    img_id = '1.' + '_'.join(filename.split('_')[:-2]) + '.png'
                    is_rg_exists = img_id in self.rg_embeddings
                    if is_rg_exists:
                        self.list_files_src.append(os.path.join(root_dir, filename))
                        self.list_files_tar.append(corset_path)
                        self.list_angles.append(np.pi * float('.'.join(filename.split('.')[:-1]).split('_')[2]) / 180)

    def __len__(self):
        return len(self.list_files_src)

    def __getitem__(self, index):
        # Идентификатор изоражения для привязки к рентгену.
        img_id = '1.' + '_'.join(self.list_files_src[index].split('/')[-1].split('_')[:-2]) + '.png'

        input_image = np.array(Image.open(self.list_files_src[index])) # Человек.
        target_image = np.array(Image.open(self.list_files_tar[index])) # Корсет.

        if self.is_train and config.USE_SHIFT_INTENSE_AUG:
            random_shift = randint(1, 10)
            input_image = np.where(input_image != 0, input_image + random_shift, input_image)
            target_image = np.where(target_image != 0, target_image + random_shift, target_image)

        augmentations = config.train_transform(image=input_image, image0=target_image) if self.is_train \
            else config.common_transform(image=input_image, image0=target_image)

        input_image = augmentations["image"]
        target_image = augmentations["image0"]
        flipped = False
        if self.is_train:
            if config.USE_FLIP_AUG:
                flipped = randint(0, 1) > 0
                if flipped:
                    input_image = input_image[:,-1::-1]
                    target_image = target_image[:,-1::-1]

            if config.USE_SCALE_INTENSE_AUG:
                scale_factor = randint(800, 1200)/1000.
                input_image = input_image * scale_factor
                target_image = target_image * scale_factor

        input_image = config.transform_to_tensor(image=input_image)["image"]
        target_image = config.transform_to_tensor(image=target_image)["image"]

        projection_angle = self.list_angles[index]
        cos = np.cos(projection_angle)
        sin = np.sin(projection_angle)
        angle = torch.from_numpy(np.array([cos, sin]))

        # Существует ли размеченный рентген для такой картинки.
        is_rg_exists = img_id in self.rg_embeddings
        rg_angles = None
        if is_rg_exists:
            rg_angles = self.rg_embeddings[img_id]
            if flipped:
                rg_angles = -rg_angles
            rg_angles = np.concatenate((rg_angles * cos, rg_angles * sin))
        else:
            rg_angles = np.zeros((list(self.rg_embeddings.values())[0].shape[0]*2,))

        rg_angles = torch.from_numpy(rg_angles)
        is_rg_exists_tensor = torch.from_numpy(np.array((int(is_rg_exists),)))

        #      входное      целевое       угол проекции  эмбеддинг с РГ     есть ли размеченный рентген
        return input_image, target_image, angle.float(), rg_angles.float(), is_rg_exists_tensor

    @staticmethod
    def bbox(img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        return rmin, cmin, rmax - rmin, cmax - cmin