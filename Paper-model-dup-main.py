# -*- coding: utf-8 -*-

import argparse
from collections import OrderedDict
import cPickle
import os
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
from MyModels import *
from generate import *

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

	data_dir = 'data/Word-based'
	prog_name = 'Paper-model-main'
	extra_name = ['model', 'model_show']

	model_file_name = '%s/%s.%s.h5' % (data_dir, prog_name, extra_name[0])
	model_show_file_name = '%s/%s.%s.h5' % (data_dir, prog_name, extra_name[1])
	print 'Loading data'
	input_file = '%s/Wid_data_divsens/wid_list.pkl' % data_dir
	train, valid, test = load_data(input_file) if options['mode'] == 'train' else load_data(input_file, 100)
	id2v = cPickle.load(open('%s/id2v.pkl' % data_dir, 'r'))
	id2v = np.matrix(id2v)

	vocab_size = len(id2v)

	print 'vocab_size = %d' % vocab_size
	print 'id2v.shape = ' + str(id2v.shape)

	print 'Building model'
	Max_sen = 7
	Sen_len = 30
	Max_len = 210 	# Body_len
	Title_len = 15
	
	model, model_show = BiGRU_Attention_Ref_2H_AutoEncoder(id2v, Sen_len=Sen_len, Max_sen=Max_sen, Title_len=Title_len)
	# Wanted output title: no BG. ref = (BG, w1, w2, ...), out = (p_w1, p_w2, p_w3, ...), label = (w1, w2, w3)
	# Otherwise the model only needs to copy input to output

	print 'model done'
	model.summary()
	plot_model(model, to_file='model.png')

	train_n = len(train[0])
	valid_n = len(valid[0])
	test_n = len(test[0])
	print "%d train examples" % train_n
	print "%d valid examples" % valid_n
	print "%d test examples" % test_n

	t = prepare_data_2d(train[0], Title_len + 1)[0]   		# Train set Titles
	b = prepare_data_3dto2d(train[1], Max_len)[0]   		# Train set Bodies
	v_t = prepare_data_2d(valid[0], Title_len + 1)[0]
	v_b = prepare_data_3dto2d(valid[1], Max_len)[0]
	ts_t = prepare_data_2d(test[0], Title_len + 1)[0]
	ts_b = prepare_data_3dto2d(test[1], Max_len)[0]
	# shape = (batch, words)

	# t_onehot = to_onehot_2d(t, vocab_size)
	# v_t_onehot = to_onehot_2d(v_t, vocab_size)
	# shape = (batch, words, vocab_size)

	def get_input_data(body_list, title_list, start_i=None, end_i=None, Title_len=Title_len):
		# body|title_list shape = (batch, timesteps)
		# Return: [[b0, t0, 0], [b1, t1, 0], ..., [b0, t0, 1], [b1, t1, 1], ..., [b0, t0, Title_len-1], ...]

		if start_i == None:
			start_i = 0
		if end_i == None:
			end_i = len(body_list)

		_b = list(body_list[start_i : end_i]) * Title_len
		_t = list(title_list[start_i : end_i, 0 : Title_len]) * Title_len
		_wpos = []
		for j in range(Title_len):
			_wpos += [j] * (end_i - start_i)

		_b = np.array(_b).astype('int64')
		_t = np.array(_t).astype('int64')
		_wpos = np.array(_wpos).astype('int64')
		comb_list = [_b, _t, _wpos]
		return comb_list

	def get_labels(title_list, start_i=None, end_i=None, Title_len=Title_len, vocab_size=vocab_size):
		# title_list shape = (batch, timesteps)
		# Return: [t_1h[:, 0], t_1h[:, 1], ..., t_1h[:, Title_len-1]]

		if start_i == None:
			start_i = 0
		if end_i == None:
			end_i = len(title_list)

		r_list = []
		for j in range(Title_len):
			r_list += list(title_list[start_i : end_i, j+1])	# Want the distribution of word j+1 at position j

		return to_onehot_1d(np.array(r_list), vocab_size)

	if options['mode'] == 'train':
		block_size = 1000
		blocks = train_n / block_size
		v_block_size = valid_n / blocks

		# if os.path.isfile(model_file_name):
		# 	model = load_model(model_file_name, custom_objects={'Attention_2H_GRU':Attention_2H_GRU, 'm_NLL':m_NLL})
		# 	model_show = load_model(model_show_file_name, custom_objects={'Attention_2H_GRU':Attention_2H_GRU, 'm_NLL':m_NLL})
		for e in range(max_epochs):
			for i in range(0, blocks):
				print 'Block %d/%d' % (i, blocks)
				model.fit(x=get_input_data(b, t, i*block_size, (i+1)*block_size),\
						  y=get_labels(t, i*block_size, (i+1)*block_size),\
						  batch_size=batch_size,\
						  validation_data=[get_input_data(v_b, v_t, i*v_block_size, (i+1)*v_block_size), get_labels(v_t, i*v_block_size, (i+1)*v_block_size)],\
						  epochs=1)

				Saveweights(model, model_file_name)
				Saveweights(model_show, model_show_file_name)
	elif options['mode'] == 'debug':
		train_input_data = get_input_data(b, t)
		train_labels = get_labels(t)
		print 'input shape = %s' % str((train_input_data[0].shape, train_input_data[1].shape, train_input_data[2].shape))
		print 'labels shape = %s' % str(train_labels.shape)
		# model.fit(x=train_input_data,\
		# 		  y=train_labels,\
		# 		  batch_size=batch_size,\
		# 		  validation_data=[get_input_data(v_b, v_t), get_labels(v_t_onehot)],\
		# 		  epochs=1)
		model.fit(x=train_input_data,\
				  y=train_labels,\
				  batch_size=batch_size,\
				  validation_data=[train_input_data, train_labels],\
				  epochs=max_epochs)
		Saveweights(model, model_file_name)
		Saveweights(model_show, model_show_file_name)
	else:
		model = load_model(model_file_name, custom_objects={'Attention_2H_GRU':Attention_2H_GRU, 'm_NLL':m_NLL})
		model_show = load_model(model_show_file_name, custom_objects={'Attention_2H_GRU':Attention_2H_GRU, 'm_NLL':m_NLL})

	# open('model_weights.txt', 'w').write(str(model.get_weights()))
	# open('model_config.txt', 'w').write(str(model.get_config()))

	# wv = KeyedVectors.load('./data/SohuNews_w2v_CHN_300.bin')
	id2w = cPickle.load(open('%s/id2w.pkl' % data_dir, 'r'))
	w2id = cPickle.load(open('%s/w2id.pkl' % data_dir, 'r'))

	show_cnt = 1
	# train_gen_title = model.predict(b[:show_cnt])
	# valid_gen_title = model.predict(v_b[:show_cnt])
	# test_gen_title = model.predict(ts_b[:show_cnt])

	# Generate(model, ts_b[0], vocab_size, w2id, id2w, beam_size=2, n_best=2)
	# quit()

	def get_output(org_title_vec, org_body_vec, body_data, ref_data, cnt, dataset_name):
		# model_show outputs = [input_emb, encode_h1, encode_h2, ref_emb, decode_seq, step_vec, output_dstrb]
		layer_names = ['input_emb', 'encode_h1', 'encode_h2', 'ref_emb', 'decode_seq', 'step_vec', 'output_dstrb']
		cnt = min(len(org_title_vec), cnt)
		_body = list(body_data) * Title_len 		# [<cnt samples for index 0>, <cnt samples for index 1>, ...]
		_ref = list(ref_data) * Title_len 			# [<cnt samples for index 0>, <cnt samples for index 1>, ...]
		_step_id = [i for i in range(Title_len) for _ in range(cnt)]

		inputs = [np.array(_body), np.array(_ref), np.array(_step_id)]

		# print '--- inputs:'
		# print inputs

		model_output = model.predict(inputs)
		model_output = np.array(model_output).reshape(cnt, Title_len, -1)
		# shape = (cnt, Title_len, dstrb)

		_all_layer_output = model_show.predict(inputs)
		# each layer: (cnt * Title_len, **args)
		all_layer_output = []
		for l in _all_layer_output:
			all_layer_output.append(np.array(l).reshape(cnt, Title_len, *l.shape[1:]))
		# each layer: (cnt, Title_len, **args)

		for i in range(cnt):			# for each sample
			org_title = ' '.join([id2w[wid] for wid in org_title_vec[i]])
			tcf_gen_title = ' '.join([id2w[np.argmax(d)] for d in model_output[i]])
			body = '\n\n'.join([' '.join([id2w[wid] for wid in sen]) for sen in org_body_vec[i]])

			if not os.path.exists('%s/Sample-output-Keras' % data_dir):
				os.mkdir('%s/Sample-output-Keras' % data_dir)

			fout = open('%s/Sample-output-Keras/out-%s%d.txt' % (data_dir, dataset_name, i), 'w')
			fout.write(('Title:\n%s\nTeacher Forced Generated Title:\n%s\n' % (org_title, tcf_gen_title)).encode('utf-8'))
			fout.write('Distribution for each word in title:\n')
			for j in range(Title_len):
				dst = model_output[i, j]	# distribution for j-th word
				k_argm = k_argmax(dst, 10)
				fout.write('%d:\n' % (j+1))
				for k in k_argm:	# k is the word_id
					fout.write(('%s: %.6lf\n' % (id2w[k], dst[k])).encode('utf-8'))
				fout.write('\n')

			fout.write('\nOutput of each layer:\n')
			for j in range(len(layer_names)):	# for each layer
				fout.write('%s:\n' % layer_names[j])
				fout.write(str(all_layer_output[j][i]) + '\n')

			best_gen_title_list = Generate(model, body_data[i], vocab_size, w2id, id2w, Title_len=Title_len)
			fout.write('\nGenerated Titles:\n')
			for k in range(len(best_gen_title_list)):
				(t, p) = best_gen_title_list[k]
				_title = ' '.join(id2w[wid] for wid in t)
				fout.write('No. %d\n%s\n%.6f\n' % (k+1, _title.encode('utf-8'), p))
			fout.write('Content:\n%s\n' % body.encode('utf-8'))

			fout.close()

	get_output(train[0], train[1], b[:show_cnt], t[:show_cnt, :Title_len], show_cnt, 'train')
	get_output(valid[0], valid[1], v_b[:show_cnt], v_t[:show_cnt, :Title_len], show_cnt, 'valid')
	get_output(test[0], test[1], ts_b[:show_cnt], ts_t[:show_cnt, :Title_len], show_cnt, 'test')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-dim_proj', type=int, default=300, help='word embeding dimension and HLSTM number of hidden units.')
    ap.add_argument('-max_epochs', type=int, default=1, help='The maximum number of epoch to run')
    ap.add_argument('-validFreq', type=int, default=10, help='Compute the validation error after this number of update.')
    ap.add_argument('-batch_size', type=int, default=100, help='The batch size during training.')
    ap.add_argument('-valid_batch_size', type=int, default=100, help='The batch size used for validation/test set.')
    ap.add_argument('-mode', type=str, default='debug', help='"train", "test" or "debug"')

    args = vars(ap.parse_args())
    train_lstm(**args)
    
