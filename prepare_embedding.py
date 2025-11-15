import os
import pickle

import cv2
import numpy as np
from tqdm import tqdm


def undesired_objects(image):
    image = image.astype('uint8')
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats((image > 0).astype(np.uint8), connectivity=8)
    sizes = stats[:, -1]
    indices = reversed(np.argsort(sizes))
    mask = np.zeros(image.shape)

    for i in indices:
        positions = output == i
        mean = np.mean(image[positions])
        if mean >= 128:
            mask[positions] = 1
            return mask, stats[i]

    return None


def process_img(path, out_path, n_splits=7):
    threshold_delta_color = 20
    threshold_points_in_row = 5

    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    degenerate = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    delta = img.astype(np.int16) - degenerate.astype(np.int16)
    delta = np.abs(delta).astype(np.uint8)
    delta = delta[:, :, 0] + delta[:, :, 1] + delta[:, :, 2]
    delta =(delta > threshold_delta_color).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    kernel_erode = np.ones((5, 5), np.uint8)
    kernel_erode[0, 0] = 0
    kernel_erode[0, 4] = 0
    kernel_erode[4, 0] = 0
    kernel_erode[4, 4] = 0

    kernel_dilate = np.ones((11,5), np.uint8)
    kernel_dilate[0, 0] = 0
    kernel_dilate[0, 4] = 0
    kernel_dilate[10, 0] = 0
    kernel_dilate[10, 4] = 0

    delta = cv2.dilate(delta, kernel, iterations=2)
    gray[delta > 0] = 255
    gray[gray > 128] = 255
    gray = 255 - gray

    gray_for_blob_detection = gray
    for cntr in range(5):
        gray_for_blob_detection = cv2.dilate(gray_for_blob_detection, kernel_dilate, iterations=5)
        gray_for_blob_detection = cv2.erode(gray_for_blob_detection, kernel_erode, iterations=5)
        #cv2.imwrite(f'fit{cntr}.png', gray_for_blob_detection)

    mask, stats = undesired_objects(gray_for_blob_detection)
    gray = np.multiply(gray, mask).astype(np.uint8)

    gray = gray[stats[1]:stats[1] + stats[3], stats[0]: stats[0] + stats[2]]
    for cntr in range(2):
        gray = cv2.dilate(gray, kernel_dilate, iterations=5)
        gray = cv2.erode(gray, kernel_erode, iterations=3)

    binary = gray > 128
    ones = np.ones_like(binary)
    x_index = np.cumsum(ones, axis=1)
    row_mean = np.mean(np.multiply(binary,x_index), axis=1)
    row_sum = np.sum(binary, axis=1)
    # print(f'correct rows:{100 * np.mean(row_sum > threshold_points_in_row)}')
    row_mean = row_mean[row_sum > threshold_points_in_row]
    center_x = np.mean(row_mean[:row_mean.shape[0] // 20])
    arrays = np.array_split(row_mean, n_splits)
    embedding = np.array([(np.mean(arr) - center_x)/row_mean.shape[0] for arr in arrays], dtype=np.float)
    cv2.imwrite(out_path, gray.astype(np.uint8))
    return embedding


def prepare_embedding(input_dir=r"..\labels\out2\out2", output_dir=r"..\labels\out2\clear", out_path="embs.pkl"):
    """
    Вычисляет эмбединги для рентгенов из input_dir
    @param input_dir: входная директория.
    @param output_dir: выходная директория (для визуализации).
    @param out_path: pkl файл с эмбеддингами
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    dictionary = {}
    for filename in tqdm(os.listdir(input_dir)):
        in_path = os.path.join(input_dir, filename)
        out_path = os.path.join(output_dir, filename)
        embedding = process_img(in_path, out_path, 9)
        dictionary[filename] = embedding

    with open(out_path, 'wb') as f:
        pickle.dump(dictionary, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    prepare_embedding()