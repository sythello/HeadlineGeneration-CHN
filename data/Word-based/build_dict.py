import math
import numpy as np
import gzip, cPickle
import os
from gensim.models import *

rng = np.random

N_WORDS = 20000
input_dir = 'Fmt_data_divsens'
# stop_words_file = 'Stop-words-none.txt'

wv = KeyedVectors.load('SohuNews_w2v_CHN_300_seg.bin')

d = {}                          # Used words: word -> (vector, freq)
w2id_dict = {}                  # word -> id
id2w = []                       # id -> word
id2v = []                       # id -> vector

# fo = open(stop_words_file, 'r')
# stop_words = fo.readlines()
# fo.close()
# for i in range(len(stop_words)):
#     stop_words[i] = stop_words[i].strip()

def regWord(w):
    if w in wv:
        if d.has_key(w):
            d[w] = (d[w][0], d[w][1] + 1)   # Add 1 frequency
        else:
            d[w] = (wv[w], 1)               # Add new entry
    # elif len(wd) > 0 and not d.has_key(wd):
    #     _v = rng.uniform(-1, 1, 300)
    #     _v = _v / np.linalg.norm(_v)
    #     d[wd] = _v

TEST_NUM = 100
fid = 0
for dirpath, dirnames, filenames in os.walk('./' + input_dir):
    for fnm in filenames:
        fo = open('%s/%s' % (dirpath, fnm), 'r')
        lines = fo.readlines()
        fo.close()

        if len(lines) < 4:     # Malformed
            continue
        title = lines[1].decode('utf-8').strip().split(' ')
        ctnt = ' '.join([lines[i].decode('utf-8').strip() for i in range(3, len(lines))]).split(' ')

        for wd in title:
            regWord(wd)
        for wd in ctnt:
            regWord(wd)

        if fid % 5000 == 0:
            print('Progress: %d' % fid)
        fid += 1

    #     if fid >= TEST_NUM:
    #         break
    # if fid >= TEST_NUM:
    #     break

print('--- Registering words completed.')
print('size of d = ' + str(len(d)))

l = list(d.items())     # item = (word, (vec, freq))
l.sort(key=lambda x : -x[1][1])
l = l[0 : N_WORDS - 1]
d_filt = dict([(x[0], x[1][0]) for x in l]) # entry: (word : vec)

print('--- Minimum frequency: %s, %d' % (l[-1][0].encode('utf-8'), l[-1][1][1]))

w2id_dict[''] = 0
id2w.append('')
id2v.append(np.zeros(300))              # word No.0 is 'padding'

for w, v in d_filt.iteritems():
    w2id_dict[w] = len(id2w)            # New index = current len of the list
    id2w.append(w)
    id2v.append(v)

fo = open('w2id.pkl', 'wb')
cPickle.dump(w2id_dict, fo)
fo.close()
fo = open('id2w.pkl', 'wb')
cPickle.dump(id2w, fo)
fo.close()
fo = open('id2v.pkl', 'wb')
cPickle.dump(id2v, fo)
fo.close()
print('--- Output completed.')
print('size of d_filt = ' + str(len(w2id_dict)))
