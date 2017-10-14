import keras
from keras.models import *
from keras.layers import *
from keras.layers.embeddings import *
from keras.layers.recurrent import *
from keras.layers.convolutional import *
from keras.layers.wrappers import *
from keras.layers.merge import *
from keras.layers.normalization import *
from keras.optimizers import *
from keras.utils.np_utils import *
from keras import backend as K

from MyLayers import *

# y_true: [[[0, ..., 1(id=w1), ..., 0], [0, ..., 1(id=w2), ..., 0], (nwords...)], (nsamples...)]
# y_pred: [[[P(w1), P(w2), ...], [P(w1), P(w2), ...], (nwords...)], (nsamples...)]
# shape = (samples, Title_len, vocab_size)
# loss: NLL
def m_SeqNLL(y_true, y_pred):
    # nsamples = K.shape(y_true)[0]
    # slen = K.shape(y_true)[1]
    # y_true_onehot_2d = K.one_hot(K.reshape(K.cast(y_true, 'int64'), (slen * nsamples, )), vocab_size)
    # y_pred_2d = -K.log(K.reshape(y_pred, (slen * nsamples, vocab_size)))
    # loss_1d = K.batch_dot(y_true_onehot_2d, y_pred_2d, axes=1)
    # loss = K.mean(loss_1d)

    _shp = K.shape(y_true)
    y_true_2d = K.reshape(y_true, (_shp[0] * _shp[1], _shp[2]))
    y_pred_2d = K.reshape(y_pred, (_shp[0] * _shp[1], _shp[2]))

    sim_loss = K.mean(K.batch_dot(y_true_2d, -K.log(y_pred_2d + 1e-8), axes=1))
    # var_loss = K.mean(K.mean(-K.var(y_pred, axis=1)))
    return sim_loss # + 10 * var_loss

def m_NLL(y_true, y_pred):
    # shape = (batch, dim)
    sim_loss = K.mean(K.batch_dot(y_true, -K.log(y_pred + 1e-8), axes=1))
    return sim_loss

# x = [seq, id]
def index_op(x):
    seq = x[0]              # shape = (batch, timesteps, dim)
    index = K.cast(x[1], dtype='int64')
    index = K.one_hot(index, K.int_shape(seq)[1])   # shape = (batch, 1, timesteps)
    out = K.batch_dot(seq, index, axes=[1,2])       # shape = (batch, dim, 1)
    out = K.squeeze(out, axis=-1)                   # shape = (batch, dim)
    return out

def index_output_shape(input_shape):
    output_shape = (input_shape[0][0], input_shape[0][2])
    # print 'output_shape = %s' % str(output_shape)
    return output_shape

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
    model.compile(optimizer='adam', loss=m_SeqNLL)
    return model

def BiLSTM_AutoEncoder_2Hierarchy(id2v, Sen_len=10, Max_sen=10, Title_len=10):
    vocab_size = len(id2v)

    model = Sequential()
    layer_e = Embedding(input_dim=vocab_size, output_dim=300, weights=[id2v], mask_zero=False, input_length=Max_sen * Sen_len)
    layer_e.trainable = False
    model.add(layer_e)
    model.add(Reshape(target_shape=(Max_sen, Sen_len, 300)))
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
    model.compile(optimizer='adam', loss=m_SeqNLL)
    return model

def BiGRU_Attention_AutoEncoder(id2v, Body_len, Title_len, h_dim=300):
    vocab_size = len(id2v)
    input_sen = Input(shape=(Body_len,))
    L_e = Embedding(input_dim=vocab_size, output_dim=300, weights=[id2v], mask_zero=True, input_length=Body_len)
    L_e.trainable = False
    input_emb = L_e(input_sen)

    encode_vec_fw = GRU(h_dim, return_sequences=False, go_backwards=False)(input_emb)
    encode_vec_bw = GRU(h_dim, return_sequences=False, go_backwards=True)(input_emb)
    encode_vec = concatenate([encode_vec_fw, encode_vec_bw])

    t1 = core.RepeatVector(Title_len)(encode_vec)

    t2 = Attention_GRU(h_dim, return_sequences=True, go_backwards=True)([t1, input_emb])
    t2 = BatchNormalization()(t2)
    t3 = Attention_GRU(h_dim, return_sequences=True, go_backwards=True)([t2, input_emb])
    t3 = BatchNormalization()(t3)
    decode_mat = Attention_GRU(h_dim, return_sequences=True, go_backwards=True)([t3, input_emb])
    decode_mat = BatchNormalization()(decode_mat)

    output_dstrb = Dense(vocab_size, activation='softmax')(decode_mat)

    model = Model(inputs=input_sen, outputs=output_dstrb)
    model.compile(optimizer='adam', loss=m_SeqNLL)
    model_show = Model(inputs=input_sen, outputs=[input_emb, encode_vec, t1, t2, t3, decode_mat, output_dstrb])
    return model, model_show

def BiGRU_Attention_Feedback_AutoEncoder(id2v, Body_len, Title_len, h_dim=300):
    vocab_size = len(id2v)
    input_sen = Input(shape=(Body_len,))
    L_e = Embedding(input_dim=vocab_size, output_dim=300, weights=[id2v], mask_zero=True, input_length=Body_len)
    L_e.trainable = False
    input_emb = L_e(input_sen)

    encode_vec_fw = GRU(h_dim, return_sequences=False, go_backwards=False)(input_emb)
    encode_vec_bw = GRU(h_dim, return_sequences=False, go_backwards=True)(input_emb)
    encode_vec = concatenate([encode_vec_fw, encode_vec_bw])

    t1 = core.RepeatVector(Title_len)(encode_vec)

    # t2 = Attention_Feedback_GRU(h_dim, id2v=id2v, return_sequences=True, go_backwards=True)([t1, input_emb])
    # t2 = BatchNormalization()(t2)
    output_dstrb = Attention_Feedback_GRU(h_dim, id2v=id2v, return_sequences=True, go_backwards=True, hard_argmax=False)([t1, input_emb])

    model = Model(inputs=input_sen, outputs=output_dstrb)
    model.compile(optimizer='adam', loss=m_SeqNLL)
    return model

def BiGRU_Attention_Ref_AutoEncoder(id2v, Sen_len, Max_sen, Title_len, h_dim=300):
    vocab_size = len(id2v)
    Body_len = Sen_len * Max_sen

    input_sen = Input(shape=(Body_len,))
    L_e1 = Embedding(input_dim=vocab_size, output_dim=300, weights=[id2v], mask_zero=False, input_length=Body_len)
    L_e1.trainable = False
    input_emb = L_e1(input_sen)

    # Treat the body as one sequence
    encode_seq_1 = GRU(h_dim, return_sequences=True, go_backwards=False)(input_emb)
    encode_seq_2 = GRU(h_dim, return_sequences=True, go_backwards=True)(input_emb)
    encode_seq = concatenate([encode_seq_1, encode_seq_2], axis=-1)
    # shape = (batch, Body_len, 2*h_dim)

    # # Treat the body as several sequences
    # encode_vec_2_h0 = Reshape(target_shape=(Max_sen, Sen_len, h_dim))(input_emb)
    # # shape = (batch, Max_sen, Sen_len, h_dim)
    # encode_vec_2_h1 = TimeDistributed(Bidirectional(GRU(h_dim, return_sequences=False), merge_mode='concat'))(encode_vec_2_h0)
    # # shape = (batch, Max_sen, 2*h_dim)
    # encode_vec_2_h2 = Bidirectional(LSTM(h_dim, return_sequences=False), merge_mode='concat')(encode_vec_2_h1)
    # # shape = (batch, 2*h_dim)

    encode_seq = Dense(h_dim, activation='softmax')(encode_seq)
    # shape = (batch, Body_len, h_dim)

    ref_sen = Input(shape=(Title_len,))
    L_e2 = Embedding(input_dim=vocab_size, output_dim=300, weights=[id2v], mask_zero=True, input_length=Title_len)
    L_e2.trainable = False
    ref_emb = L_e2(ref_sen)

    # decode_in = concatenate([encode_seq, ref_emb], axis=-1)

    decode_seq = Attention_GRU(h_dim, return_sequences=True, go_backwards=False)([ref_emb, encode_seq])
    # shape = (batch, Title_len, h_dim)

    step_id = Input(shape=(1,))
    step_vec = Lambda(index_op, output_shape=index_output_shape)([decode_seq, step_id])

    output_dstrb = Dense(vocab_size, activation='softmax')(step_vec)

    model = Model(inputs=[input_sen, ref_sen, step_id], outputs=output_dstrb)
    model.compile(optimizer='adam', loss=m_NLL)
    model_show = Model(inputs=[input_sen, ref_sen, step_id], outputs=[input_emb, encode_seq, ref_emb, decode_seq, step_vec, output_dstrb])
    return model, model_show

def BiGRU_Attention_Ref_2H_AutoEncoder(id2v, Sen_len=30, Max_sen=7, Title_len=15, h_dim=300):
    vocab_size = len(id2v)
    Body_len = Sen_len * Max_sen

    input_sen = Input(shape=(Body_len,))
    L_e1 = Embedding(input_dim=vocab_size, output_dim=300, weights=[id2v], mask_zero=False, input_length=Body_len)
    L_e1.trainable = False
    input_emb = L_e1(input_sen)

    encoder_depth_1 = 1

    ## Treat the body as several sequences
    encode_h1 = Reshape(target_shape=(Max_sen, Sen_len, h_dim))(input_emb)
    ## shape = (batch, Max_sen, Sen_len, h_dim)
    for i in range(encoder_depth_1):
        # encode_h1_r = TimeDistributed(Bidirectional(GRU(h_dim / 2, return_sequences=True), merge_mode='concat'))(encode_h1)
        # ## shape = (batch, Max_sen, Sen_len, h_dim)
        # encode_h1_c = TimeDistributed(Conv1D(filters=h_dim, kernel_size=5, padding='same'))(encode_h1)

        # if i == 0:
        #     encode_h1 = Add()([encode_h1_r, encode_h1_c])
        # else:
        #     encode_h1 = Add()([encode_h1, encode_h1_r, encode_h1_c])
        encode_h1 = TimeDistributed(Bidirectional(GRU(h_dim, return_sequences=True), merge_mode='concat'))(encode_h1)

    encode_h1 = Dense(h_dim, activation=None)(encode_h1)
    # encode_h1_masked = Masking()(encode_h1)
    ## shape = (batch, Max_sen, Sen_len, h_dim)
    encode_h1_first = Lambda(lambda x : x[:, :, 0, :], output_shape = lambda s : (s[0], s[1], s[3]))(encode_h1)
    encode_h1_last = Lambda(lambda x : x[:, :, -1, :], output_shape = lambda s : (s[0], s[1], s[3]))(encode_h1)
    encode_h1_summ = Dense(h_dim, activation=None)(Concatenate()([encode_h1_first, encode_h1_last]))
    ## shape = (batch, Max_sen, h_dim)
    encode_h2 = Bidirectional(GRU(h_dim, return_sequences=True), merge_mode='concat')(encode_h1_summ)
    encode_h2 = Dense(h_dim, activation=None)(encode_h2)
    ## shape = (batch, Max_sen, h_dim)
    encode_h2_first = Lambda(lambda x : x[:, 0, :], output_shape = lambda s : (s[0], s[2]))(encode_h2)
    encode_h2_last = Lambda(lambda x : x[:, -1, :], output_shape = lambda s : (s[0], s[2]))(encode_h2)
    encode_h2_summ = Dense(h_dim, activation=None)(Concatenate()([encode_h2_first, encode_h2_last]))

    ## shape = (batch, h_dim)

    ref_sen = Input(shape=(Title_len,))
    L_e2 = Embedding(input_dim=vocab_size, output_dim=300, weights=[id2v], mask_zero=True, input_length=Title_len)
    L_e2.trainable = False
    ref_emb = L_e2(ref_sen)
    ## shape = (batch, Title_len, wv_dim)

    decode_seq = Attention_2H_GRU(h_dim, return_sequences=True, go_backwards=False)([ref_emb, encode_h2, encode_h1], initial_state_h=encode_h2_summ)
    ## shape = (batch, Title_len, h_dim)

    # step_id = Input(shape=(1,))
    # step_vec = Lambda(index_op, output_shape=index_output_shape)([decode_seq, step_id])
    # output_dstrb = Dense(vocab_size, activation='softmax')(step_vec)

    output_dstrb = Dense(vocab_size, activation='softmax')(decode_seq)
    ## shape = (batch, Title_len, vocab_size)

    model = Model(inputs=[input_sen, ref_sen], outputs=output_dstrb)
    model.compile(optimizer=RMSprop(lr=0.0001), loss=m_SeqNLL)
    model_show = Model(inputs=[input_sen, ref_sen], outputs=[input_emb, encode_h1, encode_h2, ref_emb, decode_seq, output_dstrb])
    return model, model_show
