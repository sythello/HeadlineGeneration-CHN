# -*- coding: utf-8 -*-

import math
import numpy as np
import cPickle
import os

input_dir = 'Fmt_data_divsens'
input_name_list = os.listdir(input_dir)
output_dir = 'Wid_data_divsens'

fo = open('w2id.pkl', 'rb')
w2id = cPickle.load(fo)
fo.close()

fid = 0
DIR_SZ = 5000

for dirpath, dirnames, filenames in os.walk('./' + input_dir):
    for fnm in filenames:
        fo = open('%s/%s' % (dirpath, fnm), 'r')
        lines = fo.readlines()
        fo.close()

        if len(lines) < 4:  # Malformed
            continue
        title = lines[1].strip().decode('utf-8')
        ctnt = [lines[i].strip().decode('utf-8') for i in range(3, len(lines))]

        title_id = []
        ctnt_id = []
        for wd in title:
            if wd not in u'。！？':
                title_id.append(w2id[wd])
        for sen in ctnt:
            sen_id = []
            for wd in sen:
                sen_id.append(w2id[wd])
            ctnt_id.append(sen_id)


        if fid % DIR_SZ == 0 and not os.path.exists('./%s/%d' % (output_dir, fid / DIR_SZ)):
            os.mkdir('./%s/%d' % (output_dir, fid / DIR_SZ))

        fout = open('./%s/%d/%d.txt' % (output_dir, fid / DIR_SZ, fid), 'w')
        cPickle.dump([title_id, ctnt_id], fout)
        fout.close()

        if fid % DIR_SZ == 0:
            print('Progress: %d' % fid)
        fid += 1