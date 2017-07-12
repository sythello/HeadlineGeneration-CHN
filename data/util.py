import numpy as np
import cPickle
import os

from keras.utils.np_utils import *

# This is data/util.py


# Input: A batch
# Return: A batch with same-sized sentences and masks, shape = (nsamples, maxlen)
def prepare_data_2d(ss, maxlen):
    x = np.zeros(shape=(len(ss), maxlen), dtype='int64')
    m = np.zeros(shape=(len(ss), maxlen))
    for i in range(len(ss)):
        slen = min(len(ss[i]), maxlen)
        x[i, 0:slen] = ss[i][0:slen]
        m[i, 0:slen] = 1
    return x, m

# Input: A batch
# Return: A batch with same-sized sentences and masks, shape = (nsamples, maxsen, maxlen)
def prepare_data_3d(ss, maxsen, maxlen):
    x = np.zeros(shape=(len(ss), maxsen, maxlen), dtype='int64')
    m = np.zeros(shape=(len(ss), maxsen, maxlen))
    for i in range(len(ss)):
        sen_cnt = min(len(ss[i]), maxsen)
        for j in range(sen_cnt):
            # print 'ss[i][j].shape = ' + str(ss[i][j].shape)
            slen = min(len(ss[i][j]), maxlen)
            x[i, j, 0:slen] = ss[i][j][0:slen]
            m[i, j, 0:slen] = 1
    return x, m

# Return: 3 batches, train, dev and test
# Each batch: [0] = title, [1] = body
# title, body: (n, words)
def load_data(path):
    full_set = [[], []]
    for dirpath, dirnames, filenames in os.walk(path):
        for fnm in filenames:
            if fnm[-4:] != '.txt':  # Not a txt file
                continue
            fin = open('%s/%s' % (dirpath, fnm), 'rb')
            datas = cPickle.load(fin)
            fin.close()
            ######
            #
            # File Format:
            # [title words], [(sen1 words), (sen2 words), ...]
            #
            ######
            full_set[0].append(np.array(datas[0]).astype('int64'))
            t = []
            for s in datas[1]:
                t.append(np.array(s).astype('int64'))
            full_set[1].append(t)

            if len(full_set[0]) >= 100000:
                break
        if len(full_set[0]) >= 100000:
            break

    train = [[],[]]
    dev = [[],[]]
    test = [[],[]]
    for i in range(len(full_set[0])):
        if i % 8 <= 5:
            train[0].append(full_set[0][i])
            train[1].append(full_set[1][i])
        elif i % 8 == 6:
            dev[0].append(full_set[0][i])
            dev[1].append(full_set[1][i])
        else:
            test[0].append(full_set[0][i])
            test[1].append(full_set[1][i])

    return train, dev, test

def Embed(mat, id2v):
    new_tensor = []
    for i in range(mat.shape[0]):           # nsamples
        c_mat = id2v[mat[i]]
        new_tensor.append(c_mat)

    new_tensor = np.array(new_tensor)
    print('new_tensor.shape = ' + str(new_tensor.shape))
    return new_tensor

def d2_onehot(sens, vocab_size):
    nsamples = sens.shape[0]
    slen = sens.shape[1]
    sens_onehot = to_categorical(sens.reshape(nsamples * slen), vocab_size).reshape(nsamples, slen, vocab_size)
    return sens_onehot

