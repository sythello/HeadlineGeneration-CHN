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
output_dir = 'Fmt_data'
news_id = 0
DIR_SZ = 5000

# ff01 = '！', ff1f = '？', ff0c = '，', 3002 = '。'

# Assume 'sen' to be in unicode
def clean_word(sen):
    new_sen = []
    chn_pt = re.compile(ur'[\u4e00-\u9fa5]')
    chars = chn_pt.findall(sen)
    return ''.join([c for c in chars if c in wv])

if not os.path.exists('./%s' % output_dir):
    os.mkdir('./%s' % output_dir)

for dirpath, dirnames, filenames in os.walk('./' + input_dir):
    for fnm in filenames:
        fin = open('%s/%s' % (dirpath, fnm), 'r')
        lines = fin.readlines()
        fin.close()
        
        # Deal with title
        hd = ''
        lid = 0
        for lid in range(1, len(lines)):
            if lines[lid].find('Content:') >= 0:
                break
            hd += clean_word(lines[lid].decode('gb18030', 'ignore')).encode('utf-8')
        
        lid += 1

        # Deal with content
        ctnt = ''
        for lid in range(lid, len(lines)):
            ctnt += clean_word(lines[lid].decode('gb18030', 'ignore')).encode('utf-8')

        if len(ctnt) <= 0 or len(hd) <= 0:  # Malformed - maybe not a news file
            continue

        # Output
        if news_id % DIR_SZ == 0 and not os.path.exists('./%s/%d' % (output_dir, news_id / DIR_SZ)):
            os.mkdir('./%s/%d' % (output_dir, news_id / DIR_SZ))

        fout = open('./%s/%d/%d.txt' % (output_dir, news_id / DIR_SZ, news_id), 'w')
        fout.write('Title:\n%s\nContent:\n%s\n' % (hd, ctnt))
        fout.close()

        if news_id % DIR_SZ == 0:
            print('Progress: %d' % news_id)

        news_id += 1

