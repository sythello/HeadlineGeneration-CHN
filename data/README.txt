1. Data preprocessing
python data_raw2fmt_divsens.py
python build_dict.py
python data_fmt2id_divsens.py

(nodivsens version is suspended)

2. Encoding
Sohu_xxx.bin: unicode
Raw-data: gb2312
Fmt-data: utf-8

3. Data Format
Raw:
Title:
xxx...(original natural language)
Content:
xxx...
xxx...
xxx...

Fmt:
Title:
xxxxxx (no punctuations, all sentences concatenated, no unknown words)
Content:
xxxxxx

Wid:
[title_word_list, body_word_list]	# in the lists are all the words' indexs (BianHao)

4. About Dictionary
id2w: An array of strings, word_id -> word
w2id: An dict, word -> word_id
id2v: An array of vectors, word_id -> word_embedding_vector
