import keras
from keras.models import *
from keras.layers import *
from keras.layers.embeddings import *
from keras.layers.recurrent import *
from keras.layers.wrappers import *
from keras.layers.merge import *
from keras.layers.normalization import *
from keras.optimizers import *
from keras.utils.np_utils import *
from keras import backend as K

from MyLayers import *

# y_true: [[[0, ..., 1(id=w1), ..., 0], [0, ..., 1(id=w2), ..., 0], (nwords...)], (nsamples...)]
# y_pred: [[[P(w1), P(w2), ...], [P(w1), P(w2), ...], (nwords...)], (nsamples...)]
# loss: NLL
def myLoss(y_true, y_pred):
    # nsamples = K.shape(y_true)[0]
    # slen = K.shape(y_true)[1]
    # y_true_onehot_2d = K.one_hot(K.reshape(K.cast(y_true, 'int64'), (slen * nsamples, )), vocab_size)
    # y_pred_2d = -K.log(K.reshape(y_pred, (slen * nsamples, vocab_size)))
    # loss_1d = K.batch_dot(y_true_onehot_2d, y_pred_2d, axes=1)
    # loss = K.mean(loss_1d)

    sim_loss = K.mean(K.mean(K.batch_dot(y_true, -K.log(y_pred + 1e-6), axes=2)))
    var_loss = K.mean(K.mean(-K.var(y_pred, axis=1)))
    return sim_loss + var_loss

def BiLSTM_AutoEncoder(id2v, Body_len=100, Title_len=10):
    vocab_size = len(id2v)

    model = Sequential()
    layer_e = Embedding(input_dim=vocab_size, output_dim=300, weights=[id2v], mask_zero=True, input_length=Body_len)
    layer_e.trainable = False
    model.add(layer_e)
    model.add(Bidirectional(LSTM(300, input_shape=(30,300), return_sequences=False), merge_mode='concat'))

    model.add(core.RepeatVector(Title_len))

    model.add(Bidirectional(LSTM(300, return_sequences=True), merge_mode='concat'))

    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(optimizer='adam', loss=myLoss)
    return model

def BiLSTM_AutoEncoder_2Hierarchy(id2v, Sen_len=10, Passage_sens=10, Title_len=10):
    vocab_size = len(id2v)

    model = Sequential()
    layer_e = Embedding(input_dim=vocab_size, output_dim=300, weights=[id2v], mask_zero=False, input_length=Passage_sens * Sen_len)
    layer_e.trainable = False
    model.add(layer_e)
    model.add(Reshape(target_shape=(Passage_sens, Sen_len, 300)))
    # shape = (batch, sens, words, 300)
    model.add(TimeDistributed(Bidirectional(LSTM(300, input_shape=(10,300), return_sequences=False), merge_mode='concat')))
    # shape = (batch, sens, 600)
    model.add(Bidirectional(LSTM(300, return_sequences=False), merge_mode='concat'))
    # shape = (batch, 600)
    model.add(core.RepeatVector(Title_len))
    # shape = (batch, Title_len, 600)
    model.add(Bidirectional(LSTM(300, return_sequences=True), merge_mode='concat'))
    # shape = (batch, Title_len, 600)
    model.add(Dense(vocab_size, activation='softmax'))
    # shape = (batch, Title_len, vocab_size)
    model.compile(optimizer='adam', loss=myLoss)
    return model

def BiGRU_Attention_AutoEncoder(id2v, Body_len, Title_len, h_dim=300, fw_init=None, bw_init=None):
    vocab_size = len(id2v)

    input_sen = Input(shape=(Body_len,))
    L_e = Embedding(input_dim=vocab_size, output_dim=300, weights=[id2v], mask_zero=True, input_length=Body_len)
    L_e.trainable = False
    input_emb = L_e(input_sen)

    encode_vec_fw = GRU(h_dim, return_sequences=False, go_backwards=False)(input_emb)
    encode_vec_bw = GRU(h_dim, return_sequences=False, go_backwards=True)(input_emb)
    encode_vec = concatenate([encode_vec_fw, encode_vec_bw])

    t1 = core.RepeatVector(Title_len)(encode_vec)

    decode_mat = Attention_GRU(h_dim, return_sequences=True, go_backwards=True)([t1, input_emb])
    decode_mat = BatchNormalization()(decode_mat)

    output_dstrb = Dense(vocab_size, activation='softmax')(decode_mat)

    model = Model(inputs=input_sen, outputs=output_dstrb)
    model.compile(optimizer='adam', loss=myLoss)
    return model


