import sys
import numpy as np
sys.path.append('/home/jason6582/sfyc/attention-tensorflow')

candidateFile = sys.argv[1]
resultFile = sys.argv[2]

from core.utils_pascal import *

word_to_idx = load_word_to_idx(data_path='/home/jason6582/sfyc/attention-tensorflow/pascaldata',\
                            split='train')
# print word_to_idx
idx_to_word = {i-3:w for w, i in word_to_idx.iteritems()}

candidate = load_pickle(candidateFile)
for i in range(20):
    # print sorted(candidate[i].keys())
    candidate[i] = [candidate[i][k] for k in sorted(candidate[i].keys())]
    #print idx_to_word[i]
    #print candidate[i]

classwise_ap =[]
mAP = []

for i in range(len(candidate)):
    temp_ap = []
    for j in range(len(candidate[i])):
        if candidate[i][j] == 1:
            temp_ap.append(float(sum(candidate[i][j:])) / float(len(candidate[i][j:])))
    for j in range(len(temp_ap)):
        temp_ap[j] = max(temp_ap[:j+1])
    #print len(temp_ap)
    classwise_ap.append(sum(temp_ap)/float(len(temp_ap)))
for i, key in enumerate(sorted(word_to_idx.keys())[3:]):
    print key + "     " +  str(classwise_ap[i])
print sum(classwise_ap)/20.0

with open(resultFile, 'w') as f:
    for i, key in enumerate(sorted(word_to_idx.keys())[3:]):
        f.write(key + "     " +  str(classwise_ap[i]) + '\n')
