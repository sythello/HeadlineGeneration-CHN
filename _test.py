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

# t = np.array([[[0,0,1],
# [0,0,1],
# [1,0,0]],

# [[0,0,1],
# [0,1,0],
# [0,1,0]]])

# p = np.array([[[0.2,0.3,0.5],
# [0.2,0.3,0.5],
# [0.5,0.2,0.3]],

# [[0.2,0.3,0.5],
# [0.2,0.5,0.3],
# [0.2,0.5,0.3]]])

t = np.array([[[0,0,1],
[1,0,0]]])

p = np.array([[[0.2,0.3,0.5],
[0.5,0.2,0.3]]])

y_true = K.variable(value=t)
y_pred = K.variable(value=p)

_shp = K.shape(y_true)
y_true = K.reshape(y_true, (_shp[0] * _shp[1], _shp[2]))
y_pred = K.reshape(y_pred, (_shp[0] * _shp[1], _shp[2]))

l1 = K.batch_dot(y_true, y_pred, axes=1)
l2 = K.mean(l1)

print K.eval(l1)
print K.eval(l2)

