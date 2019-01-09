from Model import *
from selectTrainingPairs import *
import keras
import sys
import argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


def main(args):
    print(args)
    img_shape = (384, 384, 1)  # The image shape used by the model
    anisotropy = 2.15  # The horizontal compression ratio
    crop_margin = 0.05  # The margin added around the bounding box to compensate for bounding box inaccuracy

    tagged, submmit, join = get_description()
    p2size = get_p2size(join)
    p2h = get_p2h(join)
    h2ps = get_h2ps(p2h)
    h2p = get_h2p(h2ps, p2size)
    p2bb = get_bb()

    model, branch_model, head_model = build_model(64e-5, 0)
    h2ws = get_h2ws(tagged, p2h)
    w2hs = get_w2hs(h2ws)
    train, w2ts, t2i, train_set = get_training_images(w2hs)

    histories = []
    steps = 0
    features = None
    score = None
    if args.stage == 'train':
        train(model, branch_model, head_model, w2ts, t2i, steps, features, score,
            histories, train, w2hs, train_set, h2p, p2bb, p2size)
    if args.stage == 'test':
        submit(model, branch_model, head_model, w2ts, t2i, steps, features, score,
            histories, train, w2hs, train_set, h2p, p2bb, p2size, tagged, p2h)


def train(model, branch_model, head_model, w2ts, t2i, steps, features, score,
               histories, train, w2hs, train_set, h2p, p2bb, p2size):
    if os.path.isfile('/home/zhangjie/KWhaleData/piotte/mpiotte-standard.model'):
        tmp = keras.models.load_model('/home/zhangjie/KWhaleData/piotte/mpiotte-standard.model')
        model.set_weights(tmp.get_weights())
    # else:
    print('training')
    if True:
        set_lr(model, 4e-5)
        for _ in range(1):
            w2ts, t2i, steps, features, score, histories = make_steps(5, 0.25, w2ts, t2i, steps, features, score,
                       histories, train, w2hs, train_set, h2p, p2bb, p2size, model, branch_model, head_model)
        model.save('standard_train_10epochs.model')


def submit(model, branch_model, head_model, w2ts, t2i, steps, features, score,
               histories, train, w2hs, train_set, h2p, p2bb, p2size, tagged, p2h):
    if os.path.isfile('/home/zhangjie/KaggleWhale/standard_train_10epochs.model'):
        tmp = keras.models.load_model('/home/zhangjie/KaggleWhale/standard_train_10epochs.model')
        model.set_weights(tmp.get_weights())
    new_whale = 'new_whale'

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
                    for w in h2ws[h]:
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
    h2ws = {}
    for p, w in tagged.items():
        if w != new_whale:  # Use only identified whales
            h = p2h[p]
            if h not in h2ws: h2ws[h] = []
            if w not in h2ws[h]: h2ws[h].append(w)
    known = sorted(list(h2ws.keys()))

    # Dictionary of picture indices
    h2i = {}
    for i, h in enumerate(known): h2i[h] = i

    # Evaluate the model.
    fknown = branch_model.predict_generator(FeatureGen(known), max_queue_size=20, workers=10, verbose=0)
    fsubmit = branch_model.predict_generator(FeatureGen(submit), max_queue_size=20, workers=10, verbose=0)
    score = head_model.predict_generator(ScoreGen(fknown, fsubmit), max_queue_size=20, workers=10, verbose=0)
    score = score_reshape(score, fknown, fsubmit)

    # Generate the subsmission file.
    prepare_submission(0.99, 'submission.csv')
    toc = time.time()
    print("Submission time: ", (toc - tic) / 60.)


def parse_argument(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--stage', type=str, help='train or test', default='train')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_argument(sys.argv[1:]))