import sys
import numpy as np
sys.path.append('/home/jason6582/sfyc/attention-tensorflow')

candidateFile = sys.argv[1]
resultFile = sys.argv[2]

from core.utils_pascal import *
reference = load_pickle('val.references.pkl')
merge_reference = []
for key, value in reference.iteritems():
    merge_list = []
    for label in value[0]:
        merge_list += [int(label)]
    merge_reference += [merge_list]
reference = merge_reference
# print reference
candidate = load_pickle(candidateFile)
g = open(resultFile, 'w')

word_to_idx = load_word_to_idx(data_path='/home/jason6582/sfyc/attention-tensorflow/pascaldata',\
                            split='train')
# print word_to_idx
idx_to_word = {i:w for w, i in word_to_idx.iteritems()}
refsNum = 0
cansNum = 0
correctNum = 0
classwise_num = np.zeros((3,20))
num = 0
for i in range(len(candidate)):
    refs = reference[i]
    refsNum += len(refs)
    cans = candidate[i]
    cansNum += len(cans)
    refsDict = {}
    correct = 0
    can_str = []
    ref_str = []
    for c in cans:
        can_str.append(idx_to_word[c+3])
    for r in refs:
        ref_str.append(idx_to_word[r])
    print "Instance", (i+1), ":"
    print "Predicted: ", can_str
    print "Ground truth: ", ref_str
    for c in cans:
        refsDict[c+3] = 0  # follow idx of word_to_idx (keep 0, 1, 2)
        classwise_num[0][c] += 1.0 # idx start from 0 for convenience
    for r in refs:
        refsDict[r] = 1
        # idx = word_to_idx[r]
        classwise_num[1][r-3] += 1.0
    for c in cans:
        if refsDict[c+3] == 1:
            correct += 1
            # idx = word_to_idx[c]
            classwise_num[2][c] += 1.0
    correctNum += correct
recall = float(correctNum)/float(refsNum)
precision = float(correctNum)/float(cansNum)
o_f1 = 2.0/((1.0/recall) + (1.0/precision))
# print classwise_num[2]
# print classwise_num[1]
for i in range(len(classwise_num[0])):
    if classwise_num[0][i] == 0.0:
        classwise_num[0][i] = 1.0
for i in range(len(classwise_num[1])):
    if classwise_num[1][i] == 0.0:
        classwise_num[1][i] = 1.0

recall_arr = classwise_num[2] / classwise_num[1]
precision_arr = classwise_num[2] / classwise_num[0]
c_recall = np.mean(recall_arr)
c_precision = np.mean(precision_arr)
c_f1 = 2.0/((1.0/c_recall) + (1.0/c_precision))

g.write('Total number of true label: ' + str(refsNum) + '\n')
g.write('Total number of predict label: ' + str(cansNum) + '\n')
g.write('Number of correct prediction: ' + str(correctNum) + '\n\n')
g.write('O-R: ' + str(recall) + '\n')
g.write('O-P: ' + str(precision) + '\n')
g.write('O-F1: ' + str(o_f1) + '\n')
g.write('C-R: ' + str(c_recall) + '\n')
g.write('C-P: ' + str(c_precision) + '\n')
g.write('C-F1: ' + str(c_f1) + '\n')
g.write('Average: ' + str((c_f1+o_f1)/2) + '\n')
