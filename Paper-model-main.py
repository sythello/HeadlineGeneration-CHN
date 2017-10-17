# -*- coding: utf-8 -*-

import argparse
from collections import OrderedDict
import cPickle
import os
import sys
import time
from datetime import datetime

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
set_session(tf.Session(config=config))

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
	train, valid, test = load_data(input_file) if options['mode'] == 'train' else load_data(input_file, 1000)
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

	t = prepare_data_2d(train[0], Title_len + 1)[0]   									# Train set Titles
	b = prepare_data_3d(train[1], Max_sen, Sen_len)[0].reshape(train_n, Max_len)   		# Train set Bodies
	v_t = prepare_data_2d(valid[0], Title_len + 1)[0]
	v_b = prepare_data_3d(valid[1], Max_sen, Sen_len)[0].reshape(valid_n, Max_len)
	ts_t = prepare_data_2d(test[0], Title_len + 1)[0]
	ts_b = prepare_data_3d(test[1], Max_sen, Sen_len)[0].reshape(test_n, Max_len)
	# shape = (batch, words)

	# t_onehot = to_onehot_2d(t, vocab_size)
	# v_t_onehot = to_onehot_2d(v_t, vocab_size)
	# shape = (batch, words, vocab_size)

	def get_input_data(body_list, title_list, start_i=None, end_i=None, Title_len=Title_len):
		# body|title_list shape = (batch, timesteps)
		# Return: [[bodies], [titles]]
		# Return shape = [(batch, timesteps), (batch, Title_len)]

		if start_i == None:
			start_i = 0
		if end_i == None:
			end_i = len(body_list)

		_b = body_list[start_i : end_i]
		_t = title_list[start_i : end_i, 0 : Title_len]
		comb_list = [_b, _t]
		return comb_list

	def get_labels(title_list, start_i=None, end_i=None, Title_len=Title_len, vocab_size=vocab_size):
		# title_list shape = (batch, timesteps)
		# Return: [titles_one_hot]
		# Return shape = (batch, Title_len - 1, vocab_size)

		if start_i == None:
			start_i = 0
		if end_i == None:
			end_i = len(title_list)

		return to_onehot_2d(title_list[start_i : end_i, 1 : Title_len + 1], vocab_size)

	if options['mode'] == 'train':
		block_size = 1000
		blocks = train_n / block_size
		v_block_size = valid_n / blocks

		f_log = open('train.log', 'w')

		# if os.path.isfile(model_file_name):
		# 	model = load_model(model_file_name, custom_objects={'Attention_2H_GRU':Attention_2H_GRU, 'm_NLL':m_NLL})
		# 	model_show = load_model(model_show_file_name, custom_objects={'Attention_2H_GRU':Attention_2H_GRU, 'm_NLL':m_NLL})
		for e in range(max_epochs):
			for i in range(0, blocks):
				print 'Block %d/%d' % (i + e * blocks, blocks * max_epochs)
				f_log.write('Block %d/%d\n' % (i + e * blocks, blocks * max_epochs))
				model.fit(x=get_input_data(b, t, i*block_size, (i+1)*block_size),\
						  y=get_labels(t, i*block_size, (i+1)*block_size),\
						  batch_size=batch_size,\
						  validation_data=[get_input_data(v_b, v_t, i*v_block_size, (i+1)*v_block_size), get_labels(v_t, i*v_block_size, (i+1)*v_block_size)],\
						  epochs=1)

				Saveweights(model, model_file_name)
				Saveweights(model_show, model_show_file_name)
		f_log.close()

	elif options['mode'] == 'debug':
		train_input_data = get_input_data(b, t)
		train_labels = get_labels(t)
		print 'input shape = %s' % str((train_input_data[0].shape, train_input_data[1].shape))
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
		# model = load_model(model_file_name, custom_objects={'Attention_2H_GRU':Attention_2H_GRU, 'm_SeqNLL':m_SeqNLL})
		# model_show = load_model(model_show_file_name, custom_objects={'Attention_2H_GRU':Attention_2H_GRU, 'm_SeqNLL':m_SeqNLL})
		model.set_weights(load_model(model_file_name, custom_objects={'Attention_2H_GRU':Attention_2H_GRU, 'm_SeqNLL':m_SeqNLL}).get_weights())
		model_show.set_weights(load_model(model_show_file_name, custom_objects={'Attention_2H_GRU':Attention_2H_GRU, 'm_SeqNLL':m_SeqNLL}).get_weights())

	# open('model_weights.txt', 'w').write(str(model.get_weights()))
	# open('model_config.txt', 'w').write(str(model.get_config()))

	# wv = KeyedVectors.load('./data/SohuNews_w2v_CHN_300.bin')
	id2w = cPickle.load(open('%s/id2w.pkl' % data_dir, 'r'))
	w2id = cPickle.load(open('%s/w2id.pkl' % data_dir, 'r'))

	show_cnt = 5
	# train_gen_title = model.predict(b[:show_cnt])
	# valid_gen_title = model.predict(v_b[:show_cnt])
	# test_gen_title = model.predict(ts_b[:show_cnt])

	# Generate(model, ts_b[0], vocab_size, w2id, id2w, beam_size=2, n_best=2)
	# quit()

	def get_output(body_data, ref_data, cnt, dataset_name):
		# model_show outputs = [input_emb, encode_h1, encode_h2, ref_emb, decode_seq, output_dstrb]
		layer_names = ['input_emb', 'encode_h1', 'encode_h2', 'ref_emb', 'decode_seq', 'output_dstrb']
		cnt = min(len(body_data), cnt)
		body_data = body_data[:cnt]
		ref_data = ref_data[:cnt]
		## DEBUG
		# ref_data[:, 1:] = 0

		inputs = [body_data, ref_data[:, :Title_len]]

		# print '--- inputs:'
		# print inputs

		_body = body_data.reshape(cnt, Max_sen, Sen_len)
		## shape = (cnt, Max_sen, Sen_len)
		_ref = ref_data
		## shape = (cnt, Title_len + 1)

		model_output = model.predict(inputs)
		model_output = np.array(model_output).reshape(cnt, Title_len, -1)
		# shape = (cnt, Title_len, dstrb)

		all_layer_output = model_show.predict(inputs)

		for i in range(cnt):			# for each sample
			org_title = ' '.join([id2w[wid] for wid in _ref[i]])
			tcf_gen_title = ' '.join([id2w[np.argmax(d)] for d in model_output[i]])
			body = '\n\n'.join([' '.join([id2w[wid] for wid in sen]) for sen in _body[i]])

			if not os.path.exists('%s/Sample-output-Keras' % data_dir):
				os.mkdir('%s/Sample-output-Keras' % data_dir)

			fout = open('%s/Sample-output-Keras/out-%s%d.txt' % (data_dir, dataset_name, i), 'w')
			fout.write(('Title:\n%s\nTeacher Forced Generated Title:\n%s\n' % (org_title, tcf_gen_title)).encode('utf-8'))
			ppl = 0
			fout.write('Distribution for each word in title:\n')
			for j in range(Title_len):
				dst = model_output[i, j]	# distribution for j-th word
				k_argm = k_argmax(dst, 10)
				fout.write('%d:\n' % (j+1))
				for k in k_argm:	# k is the word_id
					fout.write(('%s: %.6lf\n' % (id2w[k], dst[k])).encode('utf-8'))
				fout.write(('* %s: %.6lf\n' % (id2w[ref_data[i][j+1]], dst[ref_data[i][j+1]])).encode('utf-8'))
				ppl += -np.log(dst[ref_data[i][j+1]])
				fout.write('\n')

			fout.write('\nPerplexity = %.6lf\n' % (ppl / Title_len))

			fout.write('\nOutput of each layer:\n')
			for j in range(len(layer_names)):	# for each layer
				fout.write('%s:\n' % layer_names[j])
				fout.write(str(all_layer_output[j][i]) + '\n')

			best_gen_title_list, allstep_best_open = Generate(model, body_data[i], vocab_size, w2id, id2w, Title_len=Title_len)
			# Debug
			fout.write('\nIntermediate Titles:\n')
			for step in range(len(allstep_best_open)):
				fout.write('--- Step %d ---\n' % (step + 1))
				for k in range(len(allstep_best_open[step])):
					(t, p) = allstep_best_open[step][k]
					_title = ' '.join(id2w[wid] for wid in t)
					fout.write('No. %d\n%s\n%.6f\n' % (k+1, _title.encode('utf-8'), p))
				fout.write('--- End of Step %d ---\n' % (step + 1))

			fout.write('\nGenerated Titles:\n')
			for k in range(len(best_gen_title_list)):
				(t, p) = best_gen_title_list[k]
				_title = ' '.join(id2w[wid] for wid in t)
				fout.write('No. %d\n%s\n%.6f\n' % (k+1, _title.encode('utf-8'), p))
			fout.write('Content:\n%s\n' % body.encode('utf-8'))

			fout.close()

	get_output(b, t, show_cnt, 'train')
	get_output(v_b, v_t, show_cnt, 'valid')
	get_output(ts_b, ts_t, show_cnt, 'test')

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
    
