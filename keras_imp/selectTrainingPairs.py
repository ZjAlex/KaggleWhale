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


class TrainingData(Sequence):
    def __init__(self, score, w2ts, h2p, p2bb, p2size, train, t2i, image_shape=(384, 384, 1),steps=1000, batch_size=32):
        """
        @param score the cost matrix for the picture matching
        @param steps the number of epoch we are planning with this score matrix
        """
        super(TrainingData, self).__init__()
        self.score = -score  # Maximizing the score is the same as minimuzing -score.
        self.steps = steps
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.h2p = h2p
        self.p2bb = p2bb
        self.p2size = p2size
        self.w2ts = w2ts
        self.train = train
        for ts in self.w2ts.values():
            idxs = [t2i[t] for t in ts]
            for i in idxs:
                for j in idxs:
                    self.score[i, j] = 10000.0  # Set a large value for matching whales -- eliminates this potential pairing
        self.on_epoch_end()

    def __getitem__(self, index):
        start = self.batch_size * index
        end = min(start + self.batch_size, len(self.match) + len(self.unmatch))
        size = end - start
        assert size > 0
        a = np.zeros((size,) + self.image_shape, dtype=K.floatx())
        b = np.zeros((size,) + self.image_shape, dtype=K.floatx())
        c = np.zeros((size, 1), dtype=K.floatx())
        j = start // 2
        for i in range(0, size, 2):
            a[i, :, :, :] = read_for_training(self.match[j][0], self.h2p, self.p2size, self.p2size)
            b[i, :, :, :] = read_for_training(self.match[j][1], self.h2p, self.p2size, self.p2size)
            c[i, 0] = 1  # This is a match
            a[i + 1, :, :, :] = read_for_training(self.unmatch[j][0], self.h2p, self.p2size, self.p2size)
            b[i + 1, :, :, :] = read_for_training(self.unmatch[j][1], self.h2p, self.p2size, self.p2size)
            c[i + 1, 0] = 0  # Different whales
            j += 1
        return [a, b], c

    def on_epoch_end(self):
        if self.steps <= 0: return  # Skip this on the last epoch.
        self.steps -= 1
        self.match = []
        self.unmatch = []
        _, _, x = lapjv(self.score)  # Solve the linear assignment problem
        y = np.arange(len(x), dtype=np.int32)

        # Compute a derangement for matching whales
        for ts in self.w2ts.values():
            d = ts.copy()
            while True:
                random.shuffle(d)
                if not np.any(ts == d): break
            for ab in zip(ts, d): self.match.append(ab)

        # Construct unmatched whale pairs from the LAP solution.
        for i, j in zip(x, y):
            if i == j:
                print(self.score)
                print(x)
                print(y)
                print(i, j)
            assert i != j
            self.unmatch.append((self.train[i], self.train[j]))

        # Force a different choice for an eventual next epoch.
        self.score[x, y] = 10000.0
        self.score[y, x] = 10000.0
        random.shuffle(self.match)
        random.shuffle(self.unmatch)
        # print(len(self.match), len(train), len(self.unmatch), len(train))
        assert len(self.match) == len(self.train) and len(self.unmatch) == len(self.train)

    def __len__(self):
        return (len(self.match) + len(self.unmatch) + self.batch_size - 1) // self.batch_size


# A Keras generator to evaluate only the BRANCH MODEL
class FeatureGen(Sequence):
    def __init__(self, data, h2p, p2bb, p2size, img_shape=(384, 384, 1),batch_size=64, verbose=1):
        super(FeatureGen, self).__init__()
        self.data = data
        self.batch_size = batch_size
        self.verbose = verbose
        self.img_shape = img_shape
        self.h2p = h2p
        self.p2bb = p2bb
        self.p2size = p2size
        if self.verbose > 0:
            self.progress = tqdm(total=len(self), desc='Features')

    def __getitem__(self, index):
        start = self.batch_size * index
        size = min(len(self.data) - start, self.batch_size)
        a = np.zeros((size,) + self.img_shape, dtype=K.floatx())
        for i in range(size):
            a[i, :, :, :] = read_for_validation(self.data[start + i], self.h2p, self.p2size, self.p2size)
        if self.verbose > 0:
            self.progress.update()
            if self.progress.n >= len(self): self.progress.close()
        return a

    def __len__(self):
        return (len(self.data) + self.batch_size - 1) // self.batch_size


class ScoreGen(Sequence):
    def __init__(self, x, y=None, batch_size=2048, verbose=1):
        super(ScoreGen, self).__init__()
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.verbose = verbose
        if y is None:
            self.y = self.x
            self.ix, self.iy = np.triu_indices(x.shape[0], 1)
        else:
            self.iy, self.ix = np.indices((y.shape[0], x.shape[0]))
            self.ix = self.ix.reshape((self.ix.size,))
            self.iy = self.iy.reshape((self.iy.size,))
        self.subbatch = (len(self.x) + self.batch_size - 1) // self.batch_size
        if self.verbose > 0:
            self.progress = tqdm(total=len(self), desc='Scores')

    def __getitem__(self, index):
        start = index * self.batch_size
        end = min(start + self.batch_size, len(self.ix))
        a = self.y[self.iy[start:end], :]
        b = self.x[self.ix[start:end], :]
        if self.verbose > 0:
            self.progress.update()
            if self.progress.n >= len(self): self.progress.close()
        return [a, b]

    def __len__(self):
        return (len(self.ix) + self.batch_size - 1) // self.batch_size


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


def compute_score(branch_model, head_model, train, h2p, p2bb, p2size, verbose=1):
    """
    Compute the score matrix by scoring every pictures from the training set against every other picture O(n^2).
    """
    features = branch_model.predict_generator(FeatureGen(train, h2p, p2bb, p2size, verbose=verbose), max_queue_size=12, workers=4,
                                              verbose=0)
    score = head_model.predict_generator(ScoreGen(features, verbose=verbose), max_queue_size=12, workers=4, verbose=0)
    score = score_reshape(score, features)
    return features, score


def make_steps(step, ampl, w2ts, t2i, steps, features, score,
               histories, train, w2hs, train_set, h2p, p2bb, p2size, model, branch_model, head_model):

    #global w2ts, t2i, steps, features, score, histories

    # shuffle the training pictures
    random.shuffle(train)

    # Map whale id to the list of associated training picture hash value
    w2ts = {}
    for w, hs in w2hs.items():
        for h in hs:
            if h in train_set:
                if w not in w2ts:
                    w2ts[w] = []
                if h not in w2ts[w]:
                    w2ts[w].append(h)
    for w, ts in w2ts.items():
        w2ts[w] = np.array(ts)

    # Map training picture hash value to index in 'train' array
    t2i = {}
    for i, t in enumerate(train): t2i[t] = i

    # Compute the match score for each picture pair
    features, score = compute_score(branch_model, head_model, train, h2p, p2bb, p2size)

    # Train the model for 'step' epochs
    history = model.fit_generator(
        TrainingData(score + ampl * np.random.random_sample(size=score.shape),
                     score, w2ts, h2p, p2bb, p2size, train, t2i, steps=step, batch_size=32),
        initial_epoch=steps, epochs=steps + step, max_queue_size=12, workers=6, verbose=1).history
    steps += step

    # Collect history data
    history['epochs'] = steps
    history['ms'] = np.mean(score)
    history['lr'] = get_lr(model)
    print(history['epochs'], history['lr'], history['ms'])
    histories.append(history)
    return w2ts, t2i, steps, features, score, histories