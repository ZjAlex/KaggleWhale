import random
import numpy as np
from keras.utils import Sequence
from preprocess import *
from lap import lapjv


def get_h2ws(tagged, p2h):
    h2ws = {}
    new_whale = 'new_whale'
    for p, w in tagged.items():
        if w != new_whale:
            h = p2h[p]
            if h not in h2ws:
                h2ws[h] = []
            if w not in h2ws[h]:
                h2ws[h].append(w)
    for h, ws in h2ws.items():
        if len(ws) > 1:
            h2ws[h] = sorted(ws)
    return h2ws


def get_w2hs(h2ws):
    w2hs = {}
    for h, ws in h2ws.items():
        if len(ws) == 1:  # Use only unambiguous pictures
            w = ws[0]
            if w not in w2hs:
                w2hs[w] = []
            if h not in w2hs[w]:
                w2hs[w].append(h)
    for w, hs in w2hs.items():
        if len(hs) > 1:
            w2hs[w] = sorted(hs)
    return w2hs


def get_training_images(w2hs):
    train = []  # A list of training image ids
    for hs in w2hs.values():
        if len(hs) > 1:
            train += hs
    random.shuffle(train)
    train_set = set(train)

    w2ts = {}  # Associate the image ids from train to each whale id.
    for w, hs in w2hs.items():
        for h in hs:
            if h in train_set:
                if w not in w2ts:
                    w2ts[w] = []
                if h not in w2ts[w]:
                    w2ts[w].append(h)
    for w, ts in w2ts.items():
        w2ts[w] = np.array(ts)

    t2i = {}  # The position in train of each training image id
    for i, t in enumerate(train):
        t2i[t] = i
    return train, w2ts, t2i, train_set