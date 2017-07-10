# -*- coding: utf-8 -*-

import argparse
from collections import OrderedDict
import cPickle
import sys
import time
from datetime import datetime

import theano
from theano import config
import theano.tensor as tensor
from theano.tensor.shared_randomstreams import RandomStreams

from data.util import *
from util import *
import numpy as np
import random

from gensim.models import *

def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(use_noise,
                         (state_before *
                          trng.binomial(state_before.shape,
                                        p=0.5, n=1,
                                        dtype=state_before.dtype)),
                         state_before * 0.5)
    return proj


def init_tparams(params):
    shared_params = OrderedDict()
    train_params = OrderedDict()
    for kk, pp in params.iteritems():
        tparam = theano.shared(pp, name=kk)

        shared_params[kk] = tparam
        if kk != 'Wemb':    # Do not train Wemb!
            train_params[kk] = tparam

    return shared_params, train_params


def init_params(options):
    params = OrderedDict()

    Wemb = cPickle.load(open('./data/id2v.pkl', 'rb'))
##    Wemb.append(np.random.randn(options['dim_proj']))
    params['Wemb'] = np.array(Wemb)
    params = param_init_hlstm(options, params)

    # classifier
#    params['Uq'] = 0.01 * np.random.randn(options['dim_proj']).astype(config.floatX)
#    params['Ua'] = 0.01 * np.random.randn(options['dim_proj']).astype(config.floatX)
#    params['b'] = numpy_floatX(0)

    return params


def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(config.floatX)


def param_init_hlstm(options, params):
    for h in ['h1', 'h2']:
        params['lstm_W_' + h] = np.concatenate([ortho_weight(options['dim_proj']),
                               ortho_weight(options['dim_proj']),
                               ortho_weight(options['dim_proj']),
                               ortho_weight(options['dim_proj'])], axis=1)
        params['lstm_U_' + h] = np.concatenate([ortho_weight(options['dim_proj']),
                               ortho_weight(options['dim_proj']),
                               ortho_weight(options['dim_proj']),
                               ortho_weight(options['dim_proj'])], axis=1)
        # peephole for input, forget, and output gate
        params['lstm_W_pi_' + h] = ortho_weight(options['dim_proj'])
        params['lstm_W_pf_' + h] = ortho_weight(options['dim_proj'])
        params['lstm_W_po_' + h] = ortho_weight(options['dim_proj'])

        b = np.zeros((4 * options['dim_proj'],), dtype=config.floatX)
        # increase bias for the forget gate
        b[options['dim_proj'] : 2 * options['dim_proj']] += 1.0
        params['lstm_b_' + h] = b

#    for attl in ['a']:
#        params[attl + 'att_W'] = np.concatenate([ortho_weight(options['dim_proj']),
#                               ortho_weight(options['dim_proj']),
#                               ortho_weight(options['dim_proj']),
#                               ortho_weight(options['dim_proj'])], axis=1)

#        params[attl + 'att_cand_ph'] = 0.01 * np.random.randn(options['dim_proj'], options['dim_proj']).astype(config.floatX)
#        params[attl + 'att_cand_ch'] = 0.01 * np.random.randn(options['dim_proj'], options['dim_proj']).astype(config.floatX)
#        params[attl + 'att_cand_b'] = 0.01 * np.random.randn(options['dim_proj'],).astype(config.floatX)

#        params[attl + 'att_i_ph'] = 0.01 * np.random.randn(options['dim_proj'], options['dim_proj']).astype(config.floatX)
#        params[attl + 'att_i_ch'] = 0.01 * np.random.randn(options['dim_proj'], options['dim_proj']).astype(config.floatX)
#        params[attl + 'att_i_b'] = 0.01 * np.random.randn(options['dim_proj'],).astype(config.floatX)
    
    return params


def lstm_layer(tparams, dx, dm, options, hierarchy, h_init=None, att=None):

    def _slice(_x, n, dim):
        return _x[:, n * dim:(n + 1) * dim]

    # W, U .shape = (dim, 4*dim) -> 4 parts of shape (dim, dim)
    # x_ = W * x[t] + b
    # h_ = h[t-1]
    # preact = W * x[t] + U * h[t-1] + b
    def _step(x_, m_, h_, c_):

        preact = tensor.dot(h_, tparams['lstm_U_' + hierarchy]) + x_

        i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_proj']) + tensor.dot(c_, tparams['lstm_W_pi_' + hierarchy]))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_proj']) + tensor.dot(c_, tparams['lstm_W_pf_' + hierarchy]))
        c = tensor.tanh(_slice(preact, 3, options['dim_proj']))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_proj']) + tensor.dot(c, tparams['lstm_W_po_' + hierarchy]))
        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    act = tensor.dot(dx, tparams['lstm_W_' + hierarchy]) + tparams['lstm_b_' + hierarchy]
    if att:
        act += att
    maxlen, sn = dx.shape[0], dx.shape[1]

    if h_init == None:
        h_init = tensor.alloc(numpy_floatX(0), sn, options['dim_proj'])
    rv, up = theano.scan(_step,
                    sequences=[act, dm],
                    outputs_info=[h_init, tensor.alloc(numpy_floatX(0), sn, options['dim_proj'])],
                    name='lstm_' + hierarchy,
                    n_steps=maxlen)
    return rv[0]


##def att_layer(tparams, ph, pm, ch, attl):
##    cand = tensor.tanh(tensor.dot(ph, tparams[attl + 'att_cand_ph'])[:,None,:,:] +\
##                    tensor.dot(ch, tparams[attl + 'att_cand_ch'])[None,:,:,:] +\
##                    tparams[attl + 'att_cand_b'])
##    i = tensor.nnet.sigmoid(tensor.dot(ph, tparams[attl + 'att_i_ph'])[:,None,:,:] +\
##                    tensor.dot(ch, tparams[attl + 'att_i_ch'])[None,:,:,:] +\
##                    tparams[attl + 'att_i_b'])
##    catts = (cand * i * pm[:,None,:,None]).sum(axis=0) 
##    catts = tensor.dot(catts, tparams[attl + 'att_W'])
##    return catts

# sens1, sens2 = (maxlen, n, dim)
# Return: (n, maxlen, maxlen)
def get_mat(sens1, sens2):
    sim_mat = tensor.batched_dot(sens1.dimshuffle(1,0,2), sens2.dimshuffle(1,2,0))
    return sim_mat

def build_model(tparams, options):
    trng = RandomStreams()

    use_noise = theano.shared(1.0)

    t = tensor.matrix('t', dtype='int64')   # Title
    tm = tensor.matrix('tm')
    tmaxlen, tn = t.shape[0], t.shape[1]

    b = tensor.matrix('b', dtype='int64')   # Body  
    bm = tensor.matrix('bm')
    bmaxlen, bn = b.shape[0], b.shape[1]    # Should be that bn = tn

    # lstm1: encoder
    temb = tparams['Wemb'][t.flatten()].reshape([tmaxlen, tn, options['dim_proj']])

    bemb = tparams['Wemb'][b.flatten()].reshape([bmaxlen, bn, options['dim_proj']])
    bx = lstm_layer(tparams, bemb, bm, options, 'h1', None, None)
    ctx = bx[-1]     # context vector (for each sample) (shape = sn, dim)

    # lstm2: decoder
    ctx_f = tensor.tile(ctx, (10, 1, 1))    # assume title length to be fixed at 10
    ctx_f_m = tensor.matrix('ctx_f_m')
    t_output = lstm_layer(tparams, ctx_f, ctx_f_m, options, 'h2', None, None)

    # Evaluate: Using the CNN Similarity Model
    sim_mat = get_mat(temb, t_output)
    # sim_scores = CNN_model.model('Output', sim_mat, tensor.vector(), 'params.pkl')
    sim_scores = sim_mat.norm(2, axis=[1,2])

    cost = -tensor.mean(sim_scores)

    get_title = theano.function([b, bm, ctx_f_m], t_output)
    get_cost = theano.function([t, b, bm, ctx_f_m], cost)
    get_all = theano.function([t, b, bm, ctx_f_m], [bx, ctx, ctx_f, ctx_f_m], on_unused_input='warn')   # DEBUG

    return use_noise, t, tm, b, bm, ctx_f_m, cost, get_title, get_cost, get_all


def adadelta(tparams, grads, t, tm, b, bm, ctx_f_m, cost):
    # [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning Rate Method*, arXiv:1212.5701.

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0), name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0), name='%s_rup2' % k)
                   for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0), name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inputs=[t, b, bm, ctx_f_m], outputs=cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads, running_up2, running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([], [], updates=ru2up + param_up, name='adadelta_f_update')

    return f_grad_shared, f_update

def train_lstm(
    dim_proj,  # word embeding dimension and LSTM number of hidden units.
    max_epochs,  # The maximum number of epoch to run
    validFreq,  # Compute the validation error after this number of update.
    batch_size,  # The batch size during training.
    valid_batch_size,  # The batch size used for validation/test set.
    mode
):

    # Model options
    options = locals()
    print "model options", options

    print 'Loading data'
    train, valid, test = load_data('./data/Wid-data-Reuters')

    # train, valid = merge_split(train, valid)

    print 'Building model'
    params = init_params(options)
    shared_params, train_params = init_tparams(params)

    use_noise, t, tm, b, bm, ctx_f_m, cost, get_title, get_cost, get_all = build_model(shared_params, options)

    print 'model done'

    if mode == 'train':
        print 'grads'
        grads = tensor.grad(cost, wrt=train_params.values())

        lr = tensor.scalar(name='lr')
        print 'adadelta'
        f_grad_shared, f_update = adadelta(train_params, grads, t, tm, b, bm, ctx_f_m, cost)

        print "%d train examples" % len(train[0])
        print "%d valid examples" % len(valid[0])
        print "%d test examples" % len(test[0])

        min_cost = 0
        best_params = [p.get_value() for p in train_params.itervalues()]
        total_n = len(train[0])
        batch_size = options['batch_size']
        batch_n = len(train[0]) / batch_size

        vt, vtm = prepare_data(valid[0], 10)
        vb, vbm = prepare_data(valid[1], 30)
        vctx_f_m = np.ones((10, len(valid[0])))
        
        for eidx in range(max_epochs):
            last_e_vcost = min_cost

            # Shuffle the training set
            shuffle_idx = range(total_n)
            np.random.shuffle(shuffle_idx)
            train[0] = [train[0][i] for i in shuffle_idx]
            train[1] = [train[1][i] for i in shuffle_idx]
            
            for bidx in range(batch_n):

                use_noise.set_value(1.)

                # Select the examples for this minibatch
                t = train[0][bidx * batch_size : (bidx + 1) * batch_size]
                b = train[1][bidx * batch_size : (bidx + 1) * batch_size]

                # Get the data in np.ndarray format
                # This swap the axis!
                # Return something of shape (minibatch maxlen, n samples)
                t, tm = prepare_data(t, 10)
                b, bm = prepare_data(b, 30)
                ctx_f_m = np.ones((10, batch_size))

                cost = f_grad_shared(t, b, bm, ctx_f_m)
                print 'epoch=%d, batch_id=%d, cost=%.8f' % (eidx, bidx, cost)

                f_update()

                if np.isnan(cost) or np.isinf(cost):
                    print 'bad cost detected: ', cost
                    return 0

                if np.mod(bidx, validFreq) == 0:
                    use_noise.set_value(0.)

                    v_cost = get_cost(vt, vb, vbm, vctx_f_m)
                    print 'valid cost=%.8f' % v_cost

                    if v_cost < min_cost:
                        min_cost = v_cost
                        best_params = [p.get_value() for p in train_params.itervalues()]

            if last_e_vcost - min_cost < 0.001:     # Whole epoch almost no improve
                break

        print 'The code run for %d epochs' % (eidx + 1)
        cPickle.dump(best_params, open('main_lstm.pkl', 'w'))

    else:       # mode == 'test'
        best_params = cPickle.load(open('main_lstm.pkl', 'r'))
        for par, val in zip(train_params.values(), best_params):
            par.set_value(val)

    wv = KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300-SLIM.bin.gz', binary=True)
    id2w = cPickle.load(open('./data/id2w.pkl', 'r'))

    def get_generated_title(dataset, ds_name, cnt):
        dataset[0] = dataset[0][0:cnt]
        dataset[1] = dataset[1][0:cnt]
        t, tm = prepare_data(dataset[0], 10)
        b, bm = prepare_data(dataset[1], 30)
        cfx_f_m = np.ones((10, len(dataset[0])))
        gen_t = get_title(b, bm, cfx_f_m).transpose((1,0,2))   # (title_len, ns, dim) -> (ns, title_len, dim)

        for i in range(len(dataset[0])):
            org_title = ' '.join([id2w[wid] for wid in dataset[0][i]])
            gen_title = ' '.join([wv.most_similar(positive=[vec], topn=1)[0][0] for vec in gen_t[i]])
            ctnt = ' '.join([id2w[wid] for wid in dataset[1][i]])

            fout = open('./Sample-output-theano/out-%s%d.txt' % (ds_name, i), 'w')
            fout.write('Title:\n%s\nGenerated Title:\n%s\nContent:\n%s\n' % (org_title, gen_title, ctnt))
            fout.close()

    get_generated_title(train, 'train', 10)
    get_generated_title(valid, 'valid', 10)
    get_generated_title(test, 'test', 10)

    print '*** Finished ***'


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-dim_proj', type=int, default=300, help='word embeding dimension and HLSTM number of hidden units.')
    ap.add_argument('-max_epochs', type=int, default=10, help='The maximum number of epoch to run')
    ap.add_argument('-validFreq', type=int, default=10, help='Compute the validation error after this number of update.')
    ap.add_argument('-batch_size', type=int, default=20, help='The batch size during training.')
    ap.add_argument('-valid_batch_size', type=int, default=30, help='The batch size used for validation/test set.')
    ap.add_argument('-mode', type=str, default='train', help='"train" or "test"')

    args = vars(ap.parse_args())
    train_lstm(**args)
    
