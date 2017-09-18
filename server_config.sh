download_data_dir=''	# Directory of the big files
data_dir=''				# Directory for placing datas (will move 'repo_dir/data' to 'data_dir/sls_data')
repo_dir='.'			# (Root) Directory of this repo (should be '.' by default, no need for changing)

rm -r ${data_dir}/sls_data
mv ${repo_dir}/data ${data_dir}/sls_data
ln -s ${data_dir}/sls_data ${repo_dir}/data
ln -s ${download_data_dir}/Raw_data ${repo_dir}/data/Raw_data
ln -s ${download_data_dir}/Wid_data_divsens ${repo_dir}/data/Word-based/Wid_data_divsens
ln -s ${download_data_dir}/SohuNews_w2v_CHN_300_seg.bin.syn0.npy ${repo_dir}/data/Word-based/SohuNews_w2v_CHN_300_seg.bin.syn0.npy
ln -s ${download_data_dir}/id2v.pkl ${repo_dir}/data/Word-based/id2v.pkl
ln -s ${download_data_dir}/id2w.pkl ${repo_dir}/data/Word-based/id2w.pkl
ln -s ${download_data_dir}/w2id.pkl ${repo_dir}/data/Word-based/w2id.pkl
