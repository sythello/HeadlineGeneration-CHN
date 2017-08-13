# -*- coding: utf-8 -*-

import numpy as np
import random

from data.util import *
from util import *

def Generate(
    model,
    input_sen,
    vocab_size,
    w2id,
    id2w,
    Title_len,
    beam_size=100,
    n_best=100
):
	# input_sen is a list of wid
	# Return: a list of n_best tuples with format (sen, avg_log_p)

	print ' '.join([id2w[wid] for wid in input_sen]).encode('utf-8')

	bg_id = w2id['BG']
	ed_id = w2id['ED\n']

	init_title = [bg_id] + [0 for _ in range(Title_len - 1)]

	best_open = [(init_title, 0)]							# (partial title, sum_log_p)
	best_close = []											# (partial title, avg_log_p)

	# pred_title = init_title
	# p = 0.0

	for step in range(Title_len):

	# 	DEBUG...
	#	print 'Step %d' % step
	
	# 	ref_batch = np.array([pred_title])
	# 	in_batch = np.array([input_sen])
	# 	step_batch = np.array([step])

	# 	distr = model.predict([in_batch, ref_batch, step_batch])[0]

	# 	k_best_w = k_argmax(distr, 10)
	# 	# for wid in k_best_w:			# For k-top next words for this sentence
	# 	# 	print '------ %s: %.6f' % (id2w[wid].encode('utf-8'), distr[wid])
	# 	pred_title[step] = k_best_w[0]
	# 	p += np.log(max(distr[0], 1e-6))

	# 	# print '--- Step final:'
	# 	# print '------' + ' '.join([id2w[wid] for wid in pred_title]).encode('utf-8')

	# return [(pred_title, p / 10)]

	#	GENERATE...
		ref_batch = []
		temp = []
		for (t, p) in best_open:
			ref_batch.append(t)
		ref_batch = np.array(ref_batch, dtype='int64')
		in_batch = np.array([input_sen for _ in range(len(best_open))], dtype='int64')
		step_batch = np.array([step for _ in range(len(best_open))], dtype='int64')

		# print 'in_batch.shape = %s' % str(in_batch.shape)
		# print 'ref_batch.shape = %s' % str(ref_batch.shape)
		# print 'step_batch.shape = %s' % str(step_batch.shape)

		distr_list = model.predict([in_batch, ref_batch, step_batch])

		for i in range(len(best_open)):		# For each candidate partial sentence
			t = best_open[i][0] 			# title (partial)
			p = best_open[i][1]				# (sum of log of) probability
			d = distr_list[i] 				# distribution for the next word
			# print '---' + ' '.join([id2w[wid].strip() for wid in t]).encode('utf-8')

			k_best_w = k_argmax(d, beam_size)
			for wid in k_best_w:			# For k-top next words for this sentence
				# print '------ %s: %.6f' % (id2w[wid].strip().encode('utf-8'), d[wid])
				new_p = p + np.log(max(d[wid], 1e-6))
				new_t = t[:]
				new_t[step] = wid
				temp.append((new_t, new_p))

		temp.sort(key=lambda x : -x[1])	# sort by p (big -> small)
		best_open = []
		for (t, p) in temp:
			if t[step] == ed_id:			# ended
				best_close.append( (t, p / (step + 1)) )	# average log_p
			elif t[step] == 0: 			# error
				continue
			else:
				best_open.append( (t, p) )					# sum log_p
				if len(best_open) >= beam_size:
					break
		# print '--- Step final:'
		# for (t, p) in best_open:
		# 	print '------' + ' '.join([id2w[wid].strip() for wid in t]).encode('utf-8') + ': %.6f' % p

	for (t, p) in best_open[0:n_best]:
		best_close.append( (t + [ed_id], p / (Title_len - 1)) )		

	best_close.sort(key=lambda x : -x[1])	# sort by p (big -> small)
	return best_close[0 : n_best]


