import sys
import numpy as np
sys.path.append('/home/jason6582/sfyc/attention-tensorflow')

candidateFile = sys.argv[1]
resultFile = sys.argv[2]

from core.utils_pascal_aug import *

# word_to_idx = load_word_to_idx(data_path='/home/jason6582/sfyc/attention-tensorflow/pascaldata_aug',\
#                             split='train')
word_to_idx = {'aeroplane':0, 'bicycle':1, 'bird':2, 'boat':3, 'bottle':4, 'bus':5, 'car':6, 'cat':7, 'chair':8, \
                'cow':9, 'diningtable':10, 'dog':11, 'horse':12, 'motorbike':13, 'person':14, 'pottedplant':15, 'sheep':16, \
                'sofa':17, 'train':18, 'tvmonitor':19}
idx_to_word = {i:w for w, i in word_to_idx.iteritems()}

candidate = load_pickle(candidateFile)
for i in range(20):
    # print sorted(candidate[i].keys())
    candidate[i] = sorted(candidate[i], key=lambda tup: tup[0])
    candidate[i] = [t[1] for t in candidate[i]]

classwise_ap =[]
mAP = []

for i in range(len(candidate)):
    temp_ap = []
    for j in range(len(candidate[i])):
        if candidate[i][j] == 1:
            temp_ap.append(float(sum(candidate[i][j:])) / float(len(candidate[i][j:])))
    for j in range(len(temp_ap)):
        temp_ap[j] = max(temp_ap[:j+1])
    classwise_ap.append(sum(temp_ap)/float(len(temp_ap)))
for i, key in enumerate(sorted(word_to_idx.keys())):
    print key + "     " +  str(classwise_ap[i])
print sum(classwise_ap)/20.0

with open(resultFile, 'w') as f:
    for i, key in enumerate(sorted(word_to_idx.keys())):
        f.write(key + "     " +  str(classwise_ap[i]) + '\n')
    f.write(str( sum(classwise_ap)/20.0) )
