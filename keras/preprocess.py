import pandas as pd
import time
import os
from PIL import Image
import pickle
from tqdm import tqdm
import numpy as np
from imagehash import phash
import math

# 文件路径
TRAIN_DF = '../input/humpback-whale-identification/train.csv'
SUB_Df = '../input/humpback-whale-identification/sample_submission.csv'
TRAIN = '../input/humpback-whale-identification/train/'
TEST = '../input/humpback-whale-identification/test/'
P2H = '../input/metadata/p2h.pickle'
P2SIZE = '../input/metadata/p2size.pickle'
BB_DF = "../input/metadata/bounding_boxes.csv"


# 读取训练数据和测试数据的信息
def get_description():
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
    if os.path.isfile(P2SIZE):
        print("P2SIZE already exists!")
        with open(P2SIZE, 'rb') as f:
            p2size = pickle.load(f)
    else:
        p2size = {}
        for p in tqdm(join):
            p2size[p] = Image.open(expand_path(p)).size
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
    if os.path.isfile(P2H):
        with open(P2H) as f:
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
        for p. h in p2h.items():
            h = str(h)
            if h in h2h:
                h = h2h[h]
            p2h[p] = h
    return p2h


# 获取h2ps
def get_h2ps(p2h):
    h2ps = {}
    for p, h in p2h.items():
        if h not in h2ps:
            h2ps[h] = []
        if p not in h2ps[h]:
            h2ps[h].append(p)
    return h2ps

