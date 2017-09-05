# -*- coding: utf-8 -*-

import math
import numpy as np
import cPickle
import os

input_dir = 'Fmt_data_summ'
input_name_list = os.listdir(input_dir)
output_dir = 'Wid_data_summ'

fo = open('w2id.pkl', 'rb')
w2id = cPickle.load(fo)
fo.close()

fid = 0
DIR_SZ = 5000

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

for dirpath, dirnames, filenames in os.walk(input_dir):
    for fnm in filenames:
        fo = open('%s/%s' % (dirpath, fnm), 'r')
        lines = fo.readlines()
        fo.close()

        if len(lines) != 8:  # Malformed
            continue
        title = lines[0].decode('utf-8').strip().split(' ')
        summ_list = [lines[i].decode('utf-8').strip().split(' ') for i in range(1, len(lines))]

        title_id = []
        summ_id_list = []
        for wd in title:
            title_id.append(w2id[wd])
        for summ in summ_list:
            summ_id = []
            for w in summ:
                summ_id.append(w2id[w])
            summ_id_list.append(summ_id)

        if fid % DIR_SZ == 0 and not os.path.exists('%s/%d' % (output_dir, fid / DIR_SZ)):
            os.mkdir('%s/%d' % (output_dir, fid / DIR_SZ))

        fout = open('%s/%d/%d.txt' % (output_dir, fid / DIR_SZ, fid), 'w')
        cPickle.dump([title_id, summ_id_list], fout)
        fout.close()

        if fid % DIR_SZ == 0:
            print('Progress: %d' % fid)
        fid += 1