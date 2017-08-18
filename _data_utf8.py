import os

input_dir = './data/Raw_data/gj/0/'
output_dir = './data/UTF8_Raw_data/'

for (dirpath, dirnames, filenames) in os.walk(input_dir):
	for fname in filenames:
		f = open(input_dir + fname, 'r')
		s = f.read().decode('gb18030', 'ignore').encode('utf-8')
		f.close()
		
		f = open(output_dir + fname, 'w')
		f.write(s)
		f.close()