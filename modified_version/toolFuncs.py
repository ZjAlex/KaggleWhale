import pickle
import random
from lap import lapjv
from os.path import isfile
import numpy as np
import pandas as pd
from PIL import Image as pil_image
from keras import backend as K
from keras.preprocessing.image import img_to_array
from keras.utils import Sequence
from pandas import read_csv
from scipy.ndimage import affine_transform
from tqdm import tqdm


TRAIN_DF = '/home/zhangjie/KWhaleData/train.csv'
SUB_Df = '/home/zhangjie/KWhaleData/sample_submission.csv'
TRAIN = '/home/zhangjie/KWhaleData/train/'
TEST = '/home/zhangjie/KWhaleData/test/'
P2H = '/home/zhangjie/KWhaleData/metadata/p2h.pickle'
P2SIZE = '/home/zhangjie/KWhaleData/metadata/p2size.pickle'
BB_DF = '/home/zhangjie/KWhaleData/metadata/bounding_boxes.csv'

img_shape = (384, 384, 1)  # The image shape used by the model
anisotropy = 2.15  # The horizontal compression ratio
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
def get_p2w(tagged):
    new_whale = 'new_whale'
    p2w = {}
    for p, w in tagged.items():
        if w != new_whale:
            if p not in p2w:
                p2w[p] = []
            if w not in p2w[p]:
                p2w[p].append(w)
    return p2w


def get_w2ps(p2w):
    w2ps = {}
    for p, w in p2w.items():
        if w not in w2ps:
            w2ps[w] = []
        if p not in w2ps[w]:
            w2ps[w].append(p)
    return w2ps


def read_raw_image(p):
    img = pil_image.open(expand_path(p))
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


def read_cropped_image(p, p2size, p2bb, augment):
    """
    @param p : the name of the picture to read
    @param augment: True/False if data augmentation should be performed
    @return a numpy array with the transformed image
    """
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


def read_for_training(p, p2size, p2bb):
    return read_cropped_image(p, p2size, p2bb, True)


def read_for_validation(p, p2size, p2bb):
    return read_cropped_image(p, p2size, p2bb, False)


def split_train_test(w2ps):
    np.random.seed(44)
    train = []
    test = []
    for ps in w2ps.values():
        if len(ps) >= 8:
            np.random.shuffle(ps)
            test += ps[-3:]
            train += ps[:-3]
        elif len(ps) > 1:
            train += ps
    np.random.seed(None)
    train_set = set(train)
    test_set = set(test)
    random.shuffle(train)

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

    return train, test, train_set, test_set, w2ts, w2vs, t2i, v2i


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
    """
    Tranformed the packed matrix 'score' into a square matrix.
    @param score the packed matrix
    @param x the first image feature tensor
    @param y the second image feature tensor if different from x
    @result the square matrix
    """
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


def val_score(test, threshold, known, h2kts, score_val):
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
            h = known[j]
            if a[j] < threshold and new_whale not in s:
                pos[len(t)] += 1
                s.add(new_whale)
                t.append(new_whale)
                if len(t) == 5:
                    break
            for w in h2kts[h]:
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

