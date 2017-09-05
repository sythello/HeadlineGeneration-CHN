import os

input_dir = 'Raw_data'
output_dir = 'UTF8_Raw_data'

index = 0
DIR_SZ = 5000

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

for (dirpath, dirnames, filenames) in os.walk(input_dir):
	for fnm in filenames:
		f = open('%s/%s' % (dirpath, fnm), 'r')
		s = f.read().decode('gb18030', 'ignore').encode('utf-8')
		f.close()

		if index % DIR_SZ == 0 and not os.path.exists('%s/%d' % (output_dir, index / DIR_SZ)):
			os.mkdir('%s/%d' % (output_dir, index / DIR_SZ))
		
		f = open('%s/%d/%d.txt' % (output_dir, index / DIR_SZ, index), 'w')
		f.write(s)
		f.close()

		index += 1
		if index % DIR_SZ == 0:
			print 'Progress: %d' % index