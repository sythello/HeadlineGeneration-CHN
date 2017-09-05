# -*- coding: utf-8 -*-

import math
import numpy as np
import cPickle
import os

input_dir = 'Fmt_data_divsens'
input_name_list = os.listdir(input_dir)
output_wid_dir = 'Wid_data_divsens'
output_fmt_dir = 'Fmt_data_divsens_filt'

fo = open('w2id.pkl', 'r')
w2id = cPickle.load(fo)
fo.close()

fid = 0
DIR_SZ = 5000

if not os.path.exists(output_wid_dir):
    os.mkdir(output_wid_dir)
if not os.path.exists(output_fmt_dir):
    os.mkdir(output_fmt_dir)

for dirpath, dirnames, filenames in os.walk('./' + input_dir):
    for fnm in filenames:
        fo = open('%s/%s' % (dirpath, fnm), 'r')
        lines = fo.readlines()
        fo.close()

        if len(lines) < 4:  # Malformed
            continue
        title = lines[1].decode('utf-8').strip().split(' ')
        ctnt = [lines[i].decode('utf-8').strip().split(' ') for i in range(3, len(lines))]

        title_id = []
        title_w = []
        ctnt_id = []
        ctnt_w = []
        for w in title:
            if w2id.has_key(w):
                title_id.append(w2id[w])
                title_w.append(w)
        for sen in ctnt:
            sen_id = []
            sen_w = []
            for w in sen:
                if w2id.has_key(w):
                    sen_id.append(w2id[w])
                    sen_w.append(w)
            ctnt_id.append(sen_id)
            ctnt_w.append(' '.join(sen_w))


        if fid % DIR_SZ == 0 and not os.path.exists('%s/%d' % (output_wid_dir, fid / DIR_SZ)):
            os.mkdir('%s/%d' % (output_wid_dir, fid / DIR_SZ))
        if fid % DIR_SZ == 0 and not os.path.exists('%s/%d' % (output_fmt_dir, fid / DIR_SZ)):
            os.mkdir('%s/%d' % (output_fmt_dir, fid / DIR_SZ))

        fout = open('%s/%d/%d.txt' % (output_wid_dir, fid / DIR_SZ, fid), 'w')
        cPickle.dump([title_id, ctnt_id], fout)
        fout.close()
        fout = open('%s/%d/%d.txt' % (output_fmt_dir, fid / DIR_SZ, fid), 'w')
        fout.write('Title:\n%s\nContent:\n' % (' '.join(title_w).encode('utf-8')))
        for sen in ctnt_w:
            fout.write(sen.encode('utf-8') + '\n')
        fout.close()

        if fid % DIR_SZ == 0:
            print('Progress: %d' % fid)
        fid += 1

