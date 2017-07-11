# -*- coding: utf-8 -*-

import os
import re
from gensim.models import *

######
# TODO:
#   Eliminating '\n'
#   Loop eliminating punc_ptrn

wv = KeyedVectors.load('SohuNews_w2v_CHN_300.bin')

input_dir = 'Raw_data'
output_dir = 'Fmt_data_divsens'
news_id = 0
DIR_SZ = 5000

# ff01 = '！', ff1f = '？', ff0c = '，', 3002 = '。'

# Assume 'sen' to be in unicode
def clean_word(sen):
    new_sen = []
    chn_pt = re.compile(ur'[\u4e00-\u9fa5\uff01\uff1f\u3002]')
    chars = chn_pt.findall(sen)
    return ''.join([c for c in chars if c in wv or c in u'。！？'])

if not os.path.exists('./%s' % output_dir):
    os.mkdir('./%s' % output_dir)

for dirpath, dirnames, filenames in os.walk('./' + input_dir):
    for fnm in filenames:
        fin = open('%s/%s' % (dirpath, fnm), 'r')
        lines = fin.readlines()
        fin.close()
        
        # Deal with title
        hd = ''
        for lid in range(1, len(lines)):
            if lines[lid].find('Content:') >= 0:
                break
            hd += clean_word(lines[lid].decode('gb18030', 'ignore')).encode('utf-8')
        
        lid += 1

        # Deal with content
        ctnt = ''
        for lid in range(lid, len(lines)):
            ctnt += clean_word(lines[lid].decode('gb18030', 'ignore'))
        # ctnt = all characters and stopping punctuations (unicode)

        if len(ctnt) <= 0 or len(hd) <= 0:  # Malformed - maybe not a news file
            continue
        sens = re.split(ur'！|？|。', ctnt)
        # sens = all sentences, without punctuations (unicode)

        # Output
        if news_id % DIR_SZ == 0 and not os.path.exists('./%s/%d' % (output_dir, news_id / DIR_SZ)):
            os.mkdir('./%s/%d' % (output_dir, news_id / DIR_SZ))

        fout = open('./%s/%d/%d.txt' % (output_dir, news_id / DIR_SZ, news_id), 'w')
        fout.write('Title:\n%s\nContent:\n' % hd)
        for s in sens:
            fout.write('%s\n' % s.encode('utf-8'))
        fout.close()

        if news_id % DIR_SZ == 0:
            print('Progress: %d' % news_id)

        news_id += 1

