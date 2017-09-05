# -*- coding: utf-8 -*-

import os
import re
import cPickle

f = open('id2w.pkl', 'rb')
id2w = cPickle.load(f)
f.close()
w_set = set(id2w)

input_dir = 'Raw_data_summ'
output_dir = 'Fmt_data_summ'
DIR_SZ = 5000

# ff01 = '！', ff1f = '？', ff0c = '，', 3002 = '。'

# Assume 'text' to be in unicode
def clean(text):
    begin_mark = u'BG'
    end_mark = u'ED'

    chars = re.findall(ur'[\u4e00-\u9fa5]', text)
    chars = [begin_mark] + [c for c in chars if c in w_set] + [end_mark]   # remove UNK
    res = ' '.join(chars)
    return res

def get_parts(lines):
    parts_list = []
    c_part = ''
    for lid in range(0, len(lines)):
        line = lines[lid].decode('utf-8')
        if len(line.strip()) > 0 and line.strip()[-1] == ':':       # New part
            if c_part != '':                        # Not an empty part
                parts_list.append(c_part)
            c_part = ''
            continue
        c_part += line

    if c_part != '':
        parts_list.append(c_part)

    return parts_list

if not os.path.exists('./%s' % output_dir):
    os.mkdir('./%s' % output_dir)

news_id = 0
for dirpath, dirnames, filenames in os.walk('./' + input_dir):
    for fnm in filenames:
        if fnm[-4:] != '.txt':
            continue
        fin = open('%s/%s' % (dirpath, fnm), 'r')
        lines = fin.readlines()
        fin.close()
        
        parts_list = get_parts(lines)
        if len(parts_list) != 8:         # 1 Title + 7 summs
            continue
        for i in range(len(parts_list)):
            parts_list[i] = clean(parts_list[i])

        # Output
        if news_id % DIR_SZ == 0 and not os.path.exists('./%s/%d' % (output_dir, news_id / DIR_SZ)):
            os.mkdir('%s/%d' % (output_dir, news_id / DIR_SZ))

        fout = open('%s/%d/%d.txt' % (output_dir, news_id / DIR_SZ, news_id), 'w')
        for p in parts_list:
            fout.write('%s\n' % p.encode('utf-8'))
        fout.close()
        
        news_id += 1
        if news_id % DIR_SZ == 0:
            print('Progress: %d' % news_id)

