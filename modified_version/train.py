import time
import os
import sys
from keras.callbacks import Callback
from attention_model import build_model
from keras.utils import Sequence
from lap import lapjv
from toolFuncs import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, help='decide gpu---default: 2', default='2')
parser.add_argument('--input_path', type=str, help='input path')
parser.add_argument('--output_path', type=str, help='output path')
parser.add_argument('--stage', type=str, help='train or test----default: train', default='train')
parser.add_argument('--threshold', type=float, help='threshold to decide the new whale to insert---default: 0.99', default=0.99)
parser.add_argument('--lr', type=float, help='learning rate----default: 1e-5', default=1e-5)
parser.add_argument('--epochs', type=int, help='how many epochs to iterate---default: 1', default=1)
parser.add_argument('--steps', type=int, help='how many steps one epoch---default: 5', default=5)
parser.add_argument('--reg', type=float, help='regularization rate---default: 0.0', default=0.0)
parser.add_argument('--noise', type=float, help='random noise to decide the difficult level of the trainning pairs---default: 1.0', default=1.0)
args = parser.parse_args(sys.argv[1:])

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

tagged, submit, join = get_alldata()

p2size = get_p2size(join)

p2bb = get_p2bb()

p2ws = get_p2ws(tagged)

w2ps = get_w2ps(p2ws)

train, test, train_set, test_set, w2ts, w2vs, t2i, v2i, train_soft = split_train_test(w2ps)

w2ts_soft, w2idx, train_soft_set = get_w2idx(train_soft, w2ps)

model, branch_model, head_model = build_model(args.lr, args.reg)
new_whale = 'new_whale'

p2wts = {}
for p, w in tagged.items():
    if w != new_whale:  # Use only identified whales
        if p in train_set:
            if p not in p2wts:
                p2wts[p] = []
            if w not in p2wts[p]:
                p2wts[p].append(w)
known = sorted(list(p2wts.keys()))

# Dictionary of picture indices
kt2i = {}
for i, p in enumerate(known): kt2i[p] = i

#
# p2wts_soft = {}
# for p, w in tagged.items():
#     if w != new_whale:  # Use only identified whales
#         if p in train_soft_set:
#             if p not in p2wts_soft:
#                 p2wts_soft[p] = []
#             if w not in p2wts_soft[p]:
#                 p2wts_soft[p].append(w)
# known_soft = sorted(list(p2wts_soft.keys()))
#
# # Dictionary of picture indices
# kt2i_soft = {}
# for i, p in enumerate(known_soft): kt2i_soft[p] = i


class TestingData(Sequence):
    def __init__(self, batch_size=64):
        super(TestingData, self).__init__()
        np.random.seed(10)
        self.score = -1 * np.random.random_sample(size=(len(test), len(test)))
        np.random.seed(None)
        self.batch_size = batch_size
        for vs in w2vs.values():
            idxs = [v2i[v] for v in vs]
            for i in idxs:
                for j in idxs:
                    self.score[
                        i, j] = 10000.0  # Set a large value for matching whales -- eliminates this potential pairing
        self.get_test_data()

    def __getitem__(self, index):
        start = self.batch_size * index
        end = min(start + self.batch_size, len(self.match) + len(self.unmatch))
        size = end - start
        assert size > 0
        a = np.zeros((size,) + img_shape, dtype=K.floatx())
        b = np.zeros((size,) + img_shape, dtype=K.floatx())
        c = np.zeros((size, 1), dtype=K.floatx())
        d = np.zeros((size,) + img_shape, dtype=K.floatx())
        e = np.zeros((size, 5004), dtype=K.floatx())
        j = start // 2
        for i in range(0, size, 2):
            a[i, :, :, :] = read_for_validation(self.match[j][0], p2size, p2bb)
            b[i, :, :, :] = read_for_validation(self.match[j][1], p2size, p2bb)
            c[i, 0] = 1  # This is a match
            a[i + 1, :, :, :] = read_for_validation(self.unmatch[j][0], p2size, p2bb)
            b[i + 1, :, :, :] = read_for_validation(self.unmatch[j][1], p2size, p2bb)
            c[i + 1, 0] = 0  # Different whales
            j += 1
        for i in range(size):
            d[i, :, :, :] = read_for_validation(test[(start + i) % len(test)], p2size, p2bb)
            e[i, w2idx[p2ws[test[(start + i) % len(test)]][0]]] = 1
        return [a, b, d], [c, e]

    def get_test_data(self):
        self.match = []
        self.unmatch = []
        _, _, x = lapjv(self.score)  # Solve the linear assignment problem
        y = np.arange(len(x), dtype=np.int32)

        # Compute a derangement for matching whales
        for vs in w2vs.values():
            d = vs.copy()
            while True:
                random.shuffle(d)
                if not np.any(vs == d): break
            for ab in zip(vs, d): self.match.append(ab)

        # Construct unmatched whale pairs from the LAP solution.
        for i, j in zip(x, y):
            if i == j:
                print(self.score)
                print(x)
                print(y)
                print(i, j)
            assert i != j
            self.unmatch.append((test[i], test[j]))

        # print(len(self.match), len(train), len(self.unmatch), len(train))
        assert len(self.match) == len(test) and len(self.unmatch) == len(test)

    def __len__(self):
        return (len(self.match) + len(self.unmatch) + self.batch_size - 1) // self.batch_size


class TrainingData(Sequence):
    def __init__(self, score, train_soft, steps=1000, batch_size=64):
        """
        @param score the cost matrix for the picture matching
        @param steps the number of epoch we are planning with this score matrix
        """
        super(TrainingData, self).__init__()
        self.score = -score  # Maximizing the score is the same as minimuzing -score.
        self.steps = steps
        self.batch_size = batch_size
        self.train_soft = train_soft
        for ts in w2ts.values():
            idxs = [t2i[t] for t in ts]
            for i in idxs:
                for j in idxs:
                    self.score[
                        i, j] = 10000.0  # Set a large value for matching whales -- eliminates this potential pairing
        self.on_epoch_end()

    def __getitem__(self, index):
        start = self.batch_size * index
        end = min(start + self.batch_size, len(self.match) + len(self.unmatch))
        size = end - start
        assert size > 0
        a = np.zeros((size,) + img_shape, dtype=K.floatx())
        b = np.zeros((size,) + img_shape, dtype=K.floatx())
        c = np.zeros((size, 1), dtype=K.floatx())
        d = np.zeros((size,) + img_shape, dtype=K.floatx())
        e = np.zeros((size, 5004), dtype=K.floatx())
        j = start // 2
        for i in range(0, size, 2):
            a[i, :, :, :] = read_for_training(self.match[j][0], p2size, p2bb)
            b[i, :, :, :] = read_for_training(self.match[j][1], p2size, p2bb)
            c[i, 0] = 1  # This is a match
            a[i + 1, :, :, :] = read_for_training(self.unmatch[j][0], p2size, p2bb)
            b[i + 1, :, :, :] = read_for_training(self.unmatch[j][1], p2size, p2bb)
            c[i + 1, 0] = 0  # Different whales
            j += 1
        for i in range(size):
            d[i, :, :, :] = read_for_training(self.train_soft[(start + i) % len(self.train_soft)], p2size, p2bb)
            e[i, w2idx[p2ws[self.train_soft[(start + i) % len(self.train_soft)]][0]]] = 1
        return [a, b, d], [c, e]

    def on_epoch_end(self):
        if self.steps <= 0:
            return  # Skip this on the last epoch.
        np.random.seed(None)
        np.random.shuffle(self.train_soft)
        self.steps -= 1
        self.match = []
        self.unmatch = []
        print("计算unmatch pairs")
        start0 = time.time()
        _, _, x = lapjv(self.score)  # Solve the linear assignment problem
        print("计算unmatch pairs结束,花费时间： " + str(time.time() - start0))
        y = np.arange(len(x), dtype=np.int32)

        # Compute a derangement for matching whales
        for ts in w2ts.values():
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
            self.unmatch.append((train[i], train[j]))

        # Force a different choice for an eventual next epoch.
        self.score[x, y] = 10000.0
        self.score[y, x] = 10000.0
        random.shuffle(self.match)
        random.shuffle(self.unmatch)
        # print(len(self.match), len(train), len(self.unmatch), len(train))
        assert len(self.match) == len(train) and len(self.unmatch) == len(train)

    def __len__(self):
        return (len(self.match) + len(self.unmatch) + self.batch_size - 1) // self.batch_size


# A Keras generator to evaluate only the BRANCH MODEL
class FeatureGen(Sequence):
    def __init__(self, data, batch_size=64, verbose=1):
        super(FeatureGen, self).__init__()
        self.data = data
        self.batch_size = batch_size
        self.verbose = verbose
        if self.verbose > 0: self.progress = tqdm(total=len(self), desc='Features')

    def __getitem__(self, index):
        start = self.batch_size * index
        size = min(len(self.data) - start, self.batch_size)
        a = np.zeros((size,) + img_shape, dtype=K.floatx())
        for i in range(size): a[i, :, :, :] = read_for_validation(self.data[start + i], p2size, p2bb)
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


def compute_score(verbose=1):
    """
    Compute the score matrix by scoring every pictures from the training set against every other picture O(n^2).
    """
    features = branch_model.predict_generator(FeatureGen(train, verbose=verbose), max_queue_size=12, workers=6,
                                              verbose=0)
    score = head_model.predict_generator(ScoreGen(features, verbose=verbose), max_queue_size=12, workers=6, verbose=0)
    score = score_reshape(score, features)
    return features, score


class cv_callback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 != 0:
            return
        # Evaluate the model.
        print("计算fknown")
        fknown = branch_model.predict_generator(FeatureGen(known), max_queue_size=20, workers=10, verbose=0)
        print("计算fsubmit")
        fsubmit = branch_model.predict_generator(FeatureGen(test), max_queue_size=20, workers=10, verbose=0)
        print("计算score")
        score_val = head_model.predict_generator(ScoreGen(fknown, fsubmit), max_queue_size=20, workers=10, verbose=0)
        print("计算结束")
        score_val = score_reshape(score_val, fknown, fsubmit)
        predictions = val_score(test, 0.90, known, p2wts, score_val)
        labels = [tagged[p] for p in test]

        print('cv score: ' + str(map_per_set(labels, predictions)))


def make_steps(step, ampl):
    """
    Perform training epochs
    @param step Number of epochs to perform
    @param ampl the K, the randomized component of the score matrix.
    """
    global w2ts, t2i, steps, features, score, histories

    random.shuffle(train)

    if steps == -1:
        p2wts = {}
        for p, w in tagged.items():
            if w != new_whale:  # Use only identified whales
                if p in train_set:
                    if p not in p2wts:
                        p2wts[p] = []
                    if w not in p2wts[p]:
                        p2wts[p].append(w)
        known = sorted(list(p2wts.keys()))

        # Dictionary of picture indices
        kt2i = {}
        for i, p in enumerate(known): kt2i[p] = i

        # Evaluate the model.
        print("计算fknown")
        fknown = branch_model.predict_generator(FeatureGen(known), max_queue_size=20, workers=10, verbose=0)
        print("计算fsubmit")
        fsubmit = branch_model.predict_generator(FeatureGen(test), max_queue_size=20, workers=10, verbose=0)
        print("计算score")
        score_val = head_model.predict_generator(ScoreGen(fknown, fsubmit), max_queue_size=20, workers=10, verbose=0)
        print("计算结束")
        score_val = score_reshape(score_val, fknown, fsubmit)
        predictions = val_score(test, 0.90, known, p2wts, score_val)
        labels = [tagged[p] for p in test]

        print('cv score: ' + str(map_per_set(labels, predictions)))

    # Compute the match score for each picture pair
    features, score = compute_score()

    # Train the model for 'step' epochs
    history = model.fit_generator(
        TrainingData(score + ampl * np.random.random_sample(size=score.shape), train_soft, steps=step, batch_size=64),
        initial_epoch=steps, epochs=steps + step, max_queue_size=12, workers=6,
        verbose=1).history
    steps += step

    # Collect history data
    history['epochs'] = steps
    history['ms'] = np.mean(score)
    history['lr'] = get_lr(model)
    print(history['epochs'], history['lr'], history['ms'])
    histories.append(history)


histories = []
steps = 0

if True:
    if os.path.isfile('/home/zhangjie/KWhaleData/attention_'+args.input_path+'_model_weights.h5'):
        model.load_weights('/home/zhangjie/KWhaleData/attention_'+args.input_path+'_model_weights.h5',
                           by_name=True, skip_mismatch=True, reshape=True)
    print('training')
    if args.stage == 'train':
        # epoch -> 10
        make_steps(10, 1000)
        ampl = 100.0
        for _ in range(2):
            print('noise ampl.  = ', ampl)
            make_steps(5, ampl)
            ampl = max(1.0, 100 ** -0.1 * ampl)
        model.save_weights('/home/zhangjie/KWhaleData/attention_' + args.output_path + '_model_weights.h5')
        # epoch -> 150
        for _ in range(9): make_steps(5, 1.0)
        model.save_weights('/home/zhangjie/KWhaleData/attention_' + args.output_path + '_model_weights.h5')
        # epoch -> 200
        set_lr(model, 16e-5)
        for _ in range(5): make_steps(5, 0.5)
        model.save_weights('/home/zhangjie/KWhaleData/attention_' + args.output_path + '_model_weights.h5')
        # epoch -> 240
        set_lr(model, 4e-5)
        for _ in range(4): make_steps(5, 0.25)
        # epoch -> 250
        set_lr(model, 1e-5)
        for _ in range(1): make_steps(5, 0.25)
        model.save_weights('/home/zhangjie/KWhaleData/attention_' + args.output_path + '_model_weights.h5')
        # epoch -> 300
        weights = model.get_weights()
        model, branch_model, head_model = build_model(64e-5, 0.0002)
        model.set_weights(weights)
        for _ in range(5): make_steps(5, 1.0)
        # epoch -> 350
        set_lr(model, 16e-5)
        for _ in range(5): make_steps(5, 0.5)
        model.save_weights('/home/zhangjie/KWhaleData/attention_' + args.output_path + '_model_weights.h5')
        # epoch -> 390
        set_lr(model, 4e-5)
        for _ in range(4): make_steps(5, 0.25)
        # epoch -> 400
        set_lr(model, 1e-5)
        for _ in range(1): make_steps(5, 0.25)


        # set_lr(model, args.lr)
        # for _ in range(args.epochs):
        #     make_steps(args.steps, args.noise)
        model.save_weights('/home/zhangjie/KWhaleData/attention_'+args.output_path+'_model_weights.h5')


def prepare_submission(threshold, filename):
    """
    Generate a Kaggle submission file.
    @param threshold the score given to 'new_whale'
    @param filename the submission file name
    """
    vtop = 0
    vhigh = 0
    pos = [0, 0, 0, 0, 0, 0]
    with open(filename, 'wt', newline='\n') as f:
        f.write('Image,Id\n')
        for i, p in enumerate(tqdm(submit)):
            t = []
            s = set()
            a = score[i, :]
            for j in list(reversed(np.argsort(a))):
                h = known[j]
                if a[j] < threshold and new_whale not in s:
                    pos[len(t)] += 1
                    s.add(new_whale)
                    t.append(new_whale)
                    if len(t) == 5: break;
                for w in p2ws[h]:
                    assert w != new_whale
                    if w not in s:
                        if a[j] > 1.0:
                            vtop += 1
                        elif a[j] >= threshold:
                            vhigh += 1
                        s.add(w)
                        t.append(w)
                        if len(t) == 5: break;
                if len(t) == 5: break;
            if new_whale not in s: pos[5] += 1
            assert len(t) == 5 and len(s) == 5
            f.write(p + ',' + ' '.join(t[:5]) + '\n')
    return vtop, vhigh, pos


tic = time.time()

# Evaluate the model.
fknown = branch_model.predict_generator(FeatureGen(known), max_queue_size=20, workers=10, verbose=0)
fsubmit = branch_model.predict_generator(FeatureGen(submit), max_queue_size=20, workers=10, verbose=0)
score = head_model.predict_generator(ScoreGen(fknown, fsubmit), max_queue_size=20, workers=10, verbose=0)
score = score_reshape(score, fknown, fsubmit)

# Generate the subsmission file.
prepare_submission(args.threshold, 'submission.csv')
toc = time.time()
print("Submission time: ", (toc - tic) / 60.)
