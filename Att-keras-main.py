# -*- coding: utf-8 -*-

import argparse
from collections import OrderedDict
import cPickle
import sys
import time
from datetime import datetime

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
from gensim.models import *

from data.util import *
from util import *
from MyModels import *

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
	input_dir = './data/Wid_data_divsens'
	train, valid, test = load_data(input_dir) if options['mode'] == 'train' else load_data(input_dir, 1000)
	id2v = cPickle.load(open('./data/id2v.pkl', 'r'))
	id2v = np.matrix(id2v)

	vocab_size = len(id2v)

	print 'vocab_size = %d' % vocab_size
	print 'id2v.shape = ' + str(id2v.shape)

	print 'Building model'
	Max_sen = 5
	Sen_len = 10
	Title_len = 15
	
	model = BiGRU_Attention_AutoEncoder(id2v, Max_sen * Sen_len, Title_len)

	print 'model done'
	model.summary()
	plot_model(model, to_file='model.png')

	train_n = len(train[0])
	valid_n = len(valid[0])
	test_n = len(test[0])
	print "%d train examples" % train_n
	print "%d valid examples" % valid_n
	print "%d test examples" % test_n

	t = prepare_data_2d(train[0], Title_len)[0]   						# Train set Titles
	b = prepare_data_3d(train[1], Max_sen, Sen_len)[0].reshape(train_n, -1)   	# Train set Bodies
	v_t = prepare_data_2d(valid[0], Title_len)[0]
	v_b = prepare_data_3d(valid[1], Max_sen, Sen_len)[0].reshape(valid_n, -1)
	ts_t = prepare_data_2d(test[0], Title_len)[0]
	ts_b = prepare_data_3d(test[1], Max_sen, Sen_len)[0].reshape(test_n, -1)

	t_onehot = d2_onehot(t, vocab_size)
	v_t_onehot = d2_onehot(v_t, vocab_size)

	if options['mode'] == 'train' or options['mode'] == 'debug':
		# model.load_weights('./Att-keras-main.h5')
		model.fit(b, t_onehot, batch_size=batch_size, validation_data=[v_b, v_t_onehot], epochs=max_epochs)
		if options['mode'] == 'train':
			model.save_weights('./Att-keras-main.h5')
	else:
	    model.load_weights('./Att-keras-main.h5')

	# wv = KeyedVectors.load('./data/SohuNews_w2v_CHN_300.bin')
	id2w = cPickle.load(open('./data/id2w.pkl', 'r'))

	show_cnt = 10
	train_gen_title = model.predict(b[:show_cnt])
	valid_gen_title = model.predict(v_b[:show_cnt])
	test_gen_title = model.predict(ts_b[:show_cnt])

	def get_output(org_title_vec, org_body_vec, gen_title_vec, cnt, dataset_name):
	    for i in range(min(len(org_title_vec), cnt)):	# for each document
	        org_title = ' '.join([id2w[wid] for wid in org_title_vec[i]])
	        gen_title = ' '.join([id2w[np.argmax(d)] for d in gen_title_vec[i]])
	        body = '\n'.join([' '.join([id2w[wid] for wid in sen]) for sen in org_body_vec[i]])

	        fout = open('./data/Sample-output-Keras/out-%s%d.txt' % (dataset_name, i), 'w')
	        fout.write(('Title:\n%s\nGenerated Title:\n%s\nContent:\n%s\n' % (org_title, gen_title, body)).encode('utf-8'))
	        fout.write('Generated Title Distribution:\n%s\n' % str(gen_title_vec[i]))
	        fout.write('Distribution for each word in title:\n')
	        for j in range(Title_len):
	        	dst = gen_title_vec[i][j]	# distribution for j-th word
	        	k_argm = k_argmax(dst, 10)
	        	fout.write('%d:\n' % (j+1))
	        	for k in k_argm:	# k is the word_id
	        		fout.write(('%s: %.6lf\n' % (id2w[k], dst[k])).encode('utf-8'))
	        	fout.write('\n')
	        fout.close()

	get_output(train[0], train[1], train_gen_title, show_cnt, 'train')
	get_output(valid[0], valid[1], valid_gen_title, show_cnt, 'valid')
	get_output(test[0], test[1], test_gen_title, show_cnt, 'test')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-dim_proj', type=int, default=300, help='word embeding dimension and HLSTM number of hidden units.')
    ap.add_argument('-max_epochs', type=int, default=1, help='The maximum number of epoch to run')
    ap.add_argument('-validFreq', type=int, default=10, help='Compute the validation error after this number of update.')
    ap.add_argument('-batch_size', type=int, default=20, help='The batch size during training.')
    ap.add_argument('-valid_batch_size', type=int, default=300, help='The batch size used for validation/test set.')
    ap.add_argument('-mode', type=str, default='debug', help='"train", "test" or "debug"')

    args = vars(ap.parse_args())
    train_lstm(**args)
    
