import pickle
import random
from os.path import isfile
import numpy as np
import pandas as pd
from PIL import Image as pil_image
from keras import backend as K
from pandas import read_csv
from tqdm import tqdm
from keras.preprocessing.image import apply_affine_transform


TRAIN_DF = '/home/zhangjie/KWhaleData/train.csv'
SUB_Df = '/home/zhangjie/KWhaleData/sample_submission.csv'
TRAIN = '/home/zhangjie/KWhaleData/train/'
TEST = '/home/zhangjie/KWhaleData/test/'
P2SIZE = '/home/zhangjie/KWhaleData/metadata/p2size.pickle'
BB_DF = '/home/zhangjie/KWhaleData/metadata/bounding_boxes.csv'

img_shape = (224, 224, 1)  # The image shape used by the model
crop_margin = 0.05  # The margin added around the bounding box to compensate for bounding box inaccuracy


def expand_path(p):
    if isfile(TRAIN + p):
        return TRAIN + p
    if isfile(TEST + p):
        return TEST + p
    return p


def get_alldata():
    tagged = dict([(p, w) for _, p, w in read_csv(TRAIN_DF).to_records()])
    submit = [p for _, p, _ in read_csv(SUB_Df).to_records()]
    join = list(tagged.keys()) + submit
    return tagged, submit, join


def get_p2size(join):
    if isfile(P2SIZE):
        print("P2SIZE exists.")
        with open(P2SIZE, 'rb') as f:
            p2size = pickle.load(f)
    else:
        p2size = {}
        for p in tqdm(join):
            size = pil_image.open(expand_path(p)).size
            p2size[p] = size
    return p2size


def get_p2bb():
    p2bb = pd.read_csv(BB_DF).set_index("Image")
    return p2bb


# remove new_whale
def get_p2ws(tagged):
    new_whale = 'new_whale'
    p2ws = {}
    for p, w in tagged.items():
        if w != new_whale:
            if p not in p2ws:
                p2ws[p] = []
            if w not in p2ws[p]:
                p2ws[p].append(w)
    return p2ws


def get_w2ps(p2ws):
    w2ps = {}
    for p, ws in p2ws.items():
        for w in ws:
            if w not in w2ps:
                w2ps[w] = []
            if p not in w2ps[w]:
                w2ps[w].append(p)
    return w2ps


def read_raw_image(p):
    img = pil_image.open(expand_path(p))
    return img


def get_w2idx(train_soft, w2ps):
    train_soft_set = set(train_soft)
    w2ts_soft = {}
    for w, ps in w2ps.items():
        for p in ps:
            if p in train_soft_set:
                if w not in w2ts_soft:
                    w2ts_soft[w] = []
                if p not in w2ts_soft[w]:
                    w2ts_soft[w].append(p)
    for w, ts in w2ts_soft.items():
        w2ts_soft[w] = np.array(ts)

    w2idx = {}
    for idx, w in enumerate(w2ts_soft.keys()):
        if w not in w2idx:
            w2idx[w] = idx
    return w2ts_soft, w2idx, train_soft_set



def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw, nh), pil_image.BICUBIC)
    new_image = pil_image.new('L', size)  # , (128, 128, 128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image


# 保持比例，采用keras数据增强方式
def read_cropped_image(p, p2size, p2bb, augment):
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

    img = read_raw_image(p).convert('L')

    bbox = (x0, y0, x1, y1)
    img = img.crop(bbox)
    img = letterbox_image(img, img_shape[:2])
    img = np.array(img).reshape(img_shape)
    img = img.astype(np.float32)
    if augment:
        theta = np.random.uniform(-10, 10)  # random rotation
        h, w = img.shape[0], img.shape[1]
        tx = np.random.uniform(-0.1, 0.1) * h
        ty = np.random.uniform(-0.05, 0.05) * w  # random shift
        zx, zy = np.random.uniform(0.9, 1.1, 2)  # random zoom
        shear = np.random.uniform(-10, 10)  # random shear
        img = apply_affine_transform(img, theta, tx, ty, shear, zx, zy)

    img -= np.mean(img, keepdims=True)
    img /= np.std(img, keepdims=True) + K.epsilon()
    return img


def read_for_training(p, p2size, p2bb):
    return read_cropped_image(p, p2size, p2bb, True)


def read_for_validation(p, p2size, p2bb):
    return read_cropped_image(p, p2size, p2bb, False)


def split_train_test(w2ps):
    np.random.seed(44)
    train = []
    test = []
    train_soft = []
    for ps in w2ps.values():
        if len(ps) >= 8:
            np.random.shuffle(ps)
            test += ps[-3:]
            train += ps[:-3]
            train_soft += ps[:-3]
        elif len(ps) > 1:
            train += ps
            train_soft += ps
        else:
            train_soft += ps
    np.random.seed(None)
    train_set = set(train)
    test_set = set(test)
    random.shuffle(train)
    random.shuffle(train_soft)

    w2ts = {}  # Associate the image ids from train to each whale id.
    for w, ps in w2ps.items():
        for p in ps:
            if p in train_set:
                if w not in w2ts:
                    w2ts[w] = []
                if p not in w2ts[w]:
                    w2ts[w].append(p)
    for w, ts in w2ts.items():
        w2ts[w] = np.array(ts)

    w2vs = {}  # Associate the image ids from train to each whale id.
    for w, ps in w2ps.items():
        for p in ps:
            if p in test_set:
                if w not in w2vs:
                    w2vs[w] = []
                if p not in w2vs[w]:
                    w2vs[w].append(p)
    for w, vs in w2vs.items():
        w2vs[w] = np.array(vs)

    t2i = {}  # The position in train of each training image id
    for i, t in enumerate(train):
        t2i[t] = i

    v2i = {}
    for i, v in enumerate(test):
        v2i[v] = i

    return train, test, train_set, test_set, w2ts, w2vs, t2i, v2i, train_soft


def map_per_image(label, predictions):
    try:
        return 1.0 / (predictions[:5].index(label) + 1)
    except ValueError:
        return 0.0


def map_per_set(labels, predictions):
    return np.mean([map_per_image(l, p) for l, p in zip(labels, predictions)])


def set_lr(model, lr):
    K.set_value(model.optimizer.lr, float(lr))


def get_lr(model):
    return K.get_value(model.optimizer.lr)


def score_reshape(score, x, y=None):
    if y is None:
        # When y is None, score is a packed upper triangular matrix.
        # Unpack, and transpose to form the symmetrical lower triangular matrix.
        m = np.zeros((x.shape[0], x.shape[0]), dtype=K.floatx())
        m[np.triu_indices(x.shape[0], 1)] = score.squeeze()
        m += m.transpose()
    else:
        m = np.zeros((y.shape[0], x.shape[0]), dtype=K.floatx())
        iy, ix = np.indices((y.shape[0], x.shape[0]))
        ix = ix.reshape((ix.size,))
        iy = iy.reshape((iy.size,))
        m[iy, ix] = score.squeeze()
    return m


def val_score(test, threshold, known, p2ws, score_val):
    new_whale = 'new_whale'
    vtop = 0
    vhigh = 0
    pos = [0, 0, 0, 0, 0, 0]
    predictions = []
    for i, p in enumerate(tqdm(test)):
        t = []
        s = set()
        a = score_val[i, :]
        for j in list(reversed(np.argsort(a))):
            p = known[j]
            if a[j] < threshold and new_whale not in s:
                pos[len(t)] += 1
                s.add(new_whale)
                t.append(new_whale)
                if len(t) == 5:
                    break
            for w in p2ws[p]:
                assert w != new_whale
                if w not in s:
                    if a[j] > 1.0:
                        vtop += 1
                    elif a[j] >= threshold:
                        vhigh += 1
                    s.add(w)
                    t.append(w)
                    if len(t) == 5:
                        break
            if len(t) == 5:
                break
        if new_whale not in s:
            pos[5] += 1
        assert len(t) == 5 and len(s) == 5
        predictions.append(t[:5])
    return predictions

