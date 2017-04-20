import sys
import numpy as np
sys.path.append('/home/jason6582/sfyc/attention-tensorflow')

from core.utils import *
reference = load_pickle('val.references81.pkl')
candidate = load_pickle('val.candidate.captions81__0.pkl')
g = open('norm_15.txt', 'w')

word_to_idx = load_word_to_idx(data_path='/home/jason6582/sfyc/attention-tensorflow/data', split='train')
refsNum = 0
cansNum = 0
correctNum = 0
classwise_num = np.zeros((3,81))
# for i in range(len(candidate)):
# for i in range(30):
for i in range(len(candidate)):
    refs = str(reference[i][0][:-2]).split()
    refsNum += len(refs)
    cans = str(candidate[i][:]).split()
    cansNum += len(cans)
    refsDict = {}
    correct = 0
    # print "Instance", (i+1), ":"
    # print "Predicted: ", cans
    # print "Ground truth: ", refs
    for c in cans:
        refsDict[c] = 0
        idx = word_to_idx[c]
        classwise_num[0][idx-3] += 1.0
    for r in refs:
        refsDict[r] = 1
        idx = word_to_idx[r]
        classwise_num[1][idx-3] += 1.0
    for c in cans:
        if refsDict[c] == 1:
            correct += 1
            idx = word_to_idx[c]
            classwise_num[2][idx-3] += 1.0
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
# precision_arr = np.divide(classwise_num[2], classwise_num[0])
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
