import sys
import numpy as np
sys.path.append('/home/jason6582/sfyc/attention-tensorflow/pascal2007')

from core.utils_pascal import *
reference = load_pickle('train.references.pkl')

f = open('Label_count.txt', 'wb')

word_to_idx = load_word_to_idx(data_path='/home/jason6582/sfyc/attention-tensorflow/pascal2007/pascaldata', split='train')
idx_to_word = {i:w for w, i in word_to_idx.iteritems()}

refsNum = 0
cansNum = 0
correctNum = 0
classwise_num = np.zeros(20)
num = 0
for i in range(len(reference)):
    for label in reference[i][0]:
        classwise_num[label-3] += 1
dic = {}
for i in range(20):
    dic[classwise_num[i]] = idx_to_word[i+3]

for key in reversed(sorted(dic)):
    f.write(str(dic[key])+'\n')
    f.write(str(key)+'\n\n')
