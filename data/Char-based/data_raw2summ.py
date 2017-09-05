# -*- coding: utf-8 -*-

import os

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.sum_basic import SumBasicSummarizer
from sumy.summarizers.kl import KLSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

import numpy as np

# if __name__ == '__main__':
#     url = "http://www.zsstritezuct.estranky.cz/clanky/predmety/cteni/jak-naucit-dite-spravne-cist.html"
#     parser = HtmlParser.from_url(url, Tokenizer(LANGUAGE))
#     # or for plain text files
#     # parser = PlaintextParser.from_file("document.txt", Tokenizer(LANGUAGE))
#     stemmer = Stemmer(LANGUAGE)

#     summarizer = Summarizer(stemmer)
#     summarizer.stop_words = get_stop_words(LANGUAGE)

#     for sentence in summarizer(parser.document, SENTENCES_COUNT):
#         print(sentence)

input_dir = 'Raw_data'
output_dir = 'Raw_data_summ'

LANGUAGE = "chinese"
SENTENCES_COUNT = 1
DIR_SZ = 5000
STOP_WORDS = get_stop_words(LANGUAGE)
STEMMER = Stemmer(LANGUAGE)

def get_parts(lines):
	parts_list = []
	c_part = ''
	for lid in range(0, len(lines)):
		line = lines[lid].decode('gb18030', 'ignore')
		if len(line.strip()) > 0 and line.strip()[-1] == ':':		# New part
			if c_part != '':						# Not an empty part
				parts_list.append(c_part)
			c_part = ''
			continue
		c_part += line

	if c_part != '':
		parts_list.append(c_part)

	return parts_list

def Summarize(text, Summarizer_class):
	parser = PlaintextParser(text, Tokenizer(LANGUAGE))
	summarizer = Summarizer_class(STEMMER)
	summarizer.stop_words = STOP_WORDS

	try:
		summ = ''.join([str(sen) for sen in summarizer(parser.document, SENTENCES_COUNT)])
	except np.linalg.linalg.LinAlgError:
		# print '%s LinAlgError' % str(Summarizer_class)
		summ = None
	# summ = ''.join([str(sen) for sen in summarizer(parser.document, SENTENCES_COUNT)])

	return summ

def Summarize_lead(text):
	text = text.decode('utf-8')
	for i in range(len(text)):
		if text[i] in u'。！？':
			break
	return text[:i+1].encode('utf-8')

if not os.path.exists(output_dir):
	os.mkdir(output_dir)

index = 0
for (dirpath, dirnames, filenames) in os.walk(input_dir):
	for fnm in filenames:
		if fnm[-4:] != '.txt':
			continue

		f = open('%s/%s' % (dirpath, fnm), 'r')
		lines = f.readlines()
		f.close()

		parts_list = get_parts(lines)
		if len(parts_list) < 2:
			continue					# Malformed
		title = parts_list[0].encode('utf-8')
		content = parts_list[1].encode('utf-8')

		lead = Summarize_lead(content)
		luhn = Summarize(content, LuhnSummarizer)
		lsa = Summarize(content, LsaSummarizer)
		lex_rank = Summarize(content, LexRankSummarizer)
		text_rank = Summarize(content, TextRankSummarizer)
		sum_basic = Summarize(content, SumBasicSummarizer)
		kl = Summarize(content, KLSummarizer)

		summ_list = [lead, luhn, lsa, lex_rank, text_rank, sum_basic, kl]
		summ_name_list = ['Lead', 'Luhn', 'Lsa', 'Lex_rank', 'Text_rank', 'Sum_basic', 'KL']
		if None in summ_list:	# Some summarizer failed
			continue

		# DEBUG
		# print 'Title:\n%s' % title
		# for i in range(len(summ_list)):
		# 	print '%s:\n%s\n' % (summ_name_list[i], summ_list[i])


		if index % DIR_SZ == 0 and not os.path.exists('%s/%d' % (output_dir, index / DIR_SZ)):
			os.mkdir('%s/%d' % (output_dir, index / DIR_SZ))
		
		f = open('%s/%d/%d.txt' % (output_dir, index / DIR_SZ, index), 'w')
		f.write('Title:\n%s' % title)
		for i in range(len(summ_list)):
			f.write('%s:\n%s\n' % (summ_name_list[i], summ_list[i]))
		f.close()

		index += 1
		if index % 500 == 0:
			print 'Progress: %d' % index






