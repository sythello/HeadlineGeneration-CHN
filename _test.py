import keras
from keras.models import *
from keras.layers import *
from keras.layers.embeddings import *
from keras.layers.recurrent import *
from keras.layers.wrappers import *
from keras.optimizers import *
from keras.utils import plot_model
from keras.utils.np_utils import *
from keras import backend as K

import numpy as np
import random

def testModel():
    seq = Input(shape=(15, 300))
    index = Input(shape=(1,))
    index_i = Lambda(lambda x : K.cast(x, dtype='int64'), output_shape=lambda s : s)(index)
    index_1h = Lambda(lambda x : K.one_hot(x, seq.shape[1]), output_shape=lambda s : (s[0], s[1], seq.shape[1]))(index_i)   # shape = (batch, 1, 15)
    out = Lambda(lambda x : K.batch_dot(x[0], x[1], axes=[1,2]), output_shape=lambda s : (s[0][0], s[0][2], s[1][1]))([seq, index_1h])
    M = Model(inputs=[seq, index], outputs=[index_i, index_1h, out])
    return M

M = testModel()
seq_in = np.ones((100, 15, 300))
index_in = np.ones(100)
index_i, index_1h, out = M.predict([seq_in, index_in])
print 'index_i.shape = ' + str(index_i.shape)
print 'index_1h.shape = ' + str(index_1h.shape)
print 'out.shape = ' + str(out.shape)
