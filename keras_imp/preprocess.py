import pandas as pd
import time
import os
from PIL import Image
import pickle
from tqdm import tqdm
import numpy as np
from imagehash import phash
import math
import matplotlib.pyplot as plt
import random
import keras.backend as K
from keras.preprocessing.image import img_to_array, array_to_img
from scipy.ndimage import affine_transform

# 文件路径
TRAIN_DF = '/home/zhangjie/KWhaleData/train.csv'
SUB_Df = '/home/zhangjie/KWhaleData/sample_submission.csv'
TRAIN = '/home/zhangjie/KWhaleData/train/'
TEST = '/home/zhangjie/KWhaleData/test/'
P2H = '/home/zhangjie/KWhaleData/metadata/p2h.pickle'
P2SIZE = '/home/zhangjie/KWhaleData/metadata/p2size.pickle'
BB_DF = '/home/zhangjie/KWhaleData/metadata/bounding_boxes.csv'


# 读取训练数据和测试数据的信息
def get_description():
    print("读取训练数据和测试数据的信息")
    tagged = dict([(p, w) for _, p, w in pd.read_csv(TRAIN_DF).to_records()])
    submmit = [p for _, p, _ in pd.read_csv(SUB_Df).to_records()]
    join = list(tagged.keys()) + submmit
    return tagged, submmit, join


# 得到图像的路径
def expand_path(p):
    if os.path.isfile(TRAIN + p):
        return TRAIN + p
    if os.path.isfile(TEST + p):
        return TEST + p
    return p


# 得到每个图像的大小
def get_iamges_size(join):
    print("得到每个图像的大小")
    if os.path.isfile(P2SIZE):
        print("P2SIZE already exists!")
        with open(P2SIZE, 'rb') as f:
            p2size = pickle.load(f)
    else:
        p2size = {}
        for p in tqdm(join):
            p2size[p] = Image.open(expand_path(p)).size
        data_output = open(P2SIZE, 'wb')
        pickle.dump(p2size, data_output)
    return p2size


# 判断两个phase(perceptual hash)词条是不是重复的
def match(h1, h2, h2ps):
    for p1 in h2ps[h1]:
        for p2 in h2ps[h2]:
            i1 = Image.open(expand_path(p1))
            i2 = Image.open(expand_path(p2))
            if i1.mode != i2.mode or i1.size != i2.size:
                return False
            a1 = np.array(i1)
            a1 = a1 - a1.mean()
            a1 = a1 / math.sqrt((a1**2).mean())

            a2 = np.array(i2)
            a2 = a2 - a2.mean()
            a2 = a2 / math.sqrt((a2 ** 2).mean())

            a = ((a1 - a2)**2).mean()
            if a > 0.1:
                return False
    return True


# 得到p2h（picture --- perceptual hash）
def get_p2h(join):
    print("得到p2h（picture --- perceptual hash）")
    if os.path.isfile(P2H):
        with open(P2H, 'rb') as f:
            p2h = pickle.load(f)
    else:
        p2h = {}
        for p in tqdm(join):
            img = Image.open(expand_path(p))
            h = phash(img)
            p2h[p] = h

        h2ps = {}
        for p, h in p2h.items():
            if h not in h2ps:
                h2ps[h] = []
            if p not in h2ps[h]:
                h2ps[h].append(p)
        hs = list(h2ps.keys())
        h2h = {}
        for i, h1 in enumerate(tqdm(hs)):
            for h2 in hs[:i]:
                if h1 - h2 <= 6 and match(h1, h2, h2ps):
                    s1 = str(h1)
                    s2 = str(h2)
                    if s1 < s2:
                        s1, s2 = s2, s1
                        h2h[s1] = s2
        for p, h in p2h.items():
            h = str(h)
            if h in h2h:
                h = h2h[h]
            p2h[p] = h
        data_output = open(P2H, 'wb')
        pickle.dump(p2h, data_output)
    return p2h


# 获取h2ps
def get_h2ps(p2h):
    print("获取h2ps")
    h2ps = {}
    for p, h in p2h.items():
        if h not in h2ps:
            h2ps[h] = []
        if p not in h2ps[h]:
            h2ps[h].append(p)
    return h2ps


# 展示一些重复的图片
def show_whale(imgs, per_row=2):
    n = len(imgs)
    rows = (n + per_row - 1)//per_row
    cols = min(per_row, n)
    fig, axes = plt.subplots(rows, cols, figsize=(24//per_row*cols, 24//per_row*rows))
    for ax in axes.flatten():
        ax.axis('off')
    for i, (img, ax) in enumerate(zip(imgs, axes.flatten())):
        ax.imshow(img.convert('RGB'))


def show_images(h2ps):
    for h, ps in h2ps.items():
        if len(ps) > 2:
            print('Images:', ps)
            imgs = [Image.open(expand_path(p)) for p in ps]
            show_whale(imgs, per_row=len(ps))
            break


# 为每个类别选择一个最优的图像
def prefer(ps, p2size):
    if len(ps) == 1: return ps[0]
    best_p = ps[0]
    best_s = p2size[best_p]
    for i in range(1, len(ps)):
        p = ps[i]
        s = p2size[p]
        if s[0]*s[1] > best_s[0]*best_s[1]:  # Select the image with highest resolution
            best_p = p
            best_s = s
    return best_p


def get_h2p(h2ps, p2size):
    print("获取h2p")
    h2p = {}
    for h, ps in h2ps.items():
        h2p[h] = prefer(ps, p2size)
    return h2p


# 读取boundingbox数据
def get_bb():
    print("获取boundingbox数据")
    p2bb = pd.read_csv(open(BB_DF)).set_index('Image')
    return p2bb


def read_raw_image(p):
    img = Image.open(expand_path(p))
    return img


def build_transform(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
    """
    Build a transformation matrix with the specified characteristics.
    """
    rotation = np.deg2rad(rotation)
    shear = np.deg2rad(shear)
    rotation_matrix = np.array(
        [[np.cos(rotation), np.sin(rotation), 0], [-np.sin(rotation), np.cos(rotation), 0], [0, 0, 1]])
    shift_matrix = np.array([[1, 0, height_shift], [0, 1, width_shift], [0, 0, 1]])
    shear_matrix = np.array([[1, np.sin(shear), 0], [0, np.cos(shear), 0], [0, 0, 1]])
    zoom_matrix = np.array([[1.0 / height_zoom, 0, 0], [0, 1.0 / width_zoom, 0], [0, 0, 1]])
    shift_matrix = np.array([[1, 0, -height_shift], [0, 1, -width_shift], [0, 0, 1]])
    return np.dot(np.dot(rotation_matrix, shear_matrix), np.dot(zoom_matrix, shift_matrix))


def read_cropped_image(p, h2p, p2bb, p2size, augment=True, crop_margin=0.05,
                       img_shape=(384, 384, 1), anisotropy=2.15):
    # If an image id was given, convert to filename
    if p in h2p:
        p = h2p[p]
    size_x, size_y = p2size[p]

    # Determine the region of the original image we want to capture based on the bounding box.
    row = p2bb.loc[p]
    x0, y0, x1, y1 = row['x0'], row['y0'], row['x1'], row['y1']
    dx = x1 - x0
    dy = y1 - y0
    x0 -= dx * crop_margin
    x1 += dx * crop_margin + 1
    y0 -= dy * crop_margin
    y1 += dy * crop_margin + 1
    if x0 < 0:
        x0 = 0
    if x1 > size_x:
        x1 = size_x
    if y0 < 0:
        y0 = 0
    if y1 > size_y:
        y1 = size_y
    dx = x1 - x0
    dy = y1 - y0
    if dx > dy * anisotropy:
        dy = 0.5 * (dx / anisotropy - dy)
        y0 -= dy
        y1 += dy
    else:
        dx = 0.5 * (dy * anisotropy - dx)
        x0 -= dx
        x1 += dx

    # Generate the transformation matrix
    trans = np.array([[1, 0, -0.5 * img_shape[0]], [0, 1, -0.5 * img_shape[1]], [0, 0, 1]])
    trans = np.dot(np.array([[(y1 - y0) / img_shape[0], 0, 0], [0, (x1 - x0) / img_shape[1], 0], [0, 0, 1]]), trans)
    if augment:
        trans = np.dot(build_transform(
            random.uniform(-5, 5),
            random.uniform(-5, 5),
            random.uniform(0.8, 1.0),
            random.uniform(0.8, 1.0),
            random.uniform(-0.05 * (y1 - y0), 0.05 * (y1 - y0)),
            random.uniform(-0.05 * (x1 - x0), 0.05 * (x1 - x0))
        ), trans)
    trans = np.dot(np.array([[1, 0, 0.5 * (y1 + y0)], [0, 1, 0.5 * (x1 + x0)], [0, 0, 1]]), trans)

    # Read the image, transform to black and white and comvert to numpy array
    img = read_raw_image(p).convert('L')
    img = img_to_array(img)

    # Apply affine transformation
    matrix = trans[:2, :2]
    offset = trans[:2, 2]
    img = img.reshape(img.shape[:-1])
    img = affine_transform(img, matrix, offset, output_shape=img_shape[:-1], order=1, mode='constant',
                           cval=np.average(img))
    img = img.reshape(img_shape)

    # Normalize to zero mean and unit variance
    img -= np.mean(img, keepdims=True)
    img /= np.std(img, keepdims=True) + K.epsilon()
    return img


def read_for_training(p, h2p, p2bb, p2size):
    return read_cropped_image(p, h2p, p2bb, p2size, augment=True)


def read_for_validation(p, h2p, p2bb, p2size):
    return read_cropped_image(p, h2p, p2bb, p2size, augment=False)


def test_image_transform(p, tagged, h2p, p2bb, p2size):
    imgs = [
        read_raw_image(p),
        array_to_img(read_for_validation(p, h2p, p2bb, p2size)),
        array_to_img(read_for_training(p, h2p, p2bb, p2size))
    ]
    show_whale(imgs, per_row=3)


