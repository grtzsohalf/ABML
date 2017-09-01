import sys
import numpy as np
sys.path.append('/home/jason6582/sfyc/attention-tensorflow/nus-wide')

from core.utils_nus import *
import hickle

init_pred = hickle.load('test.init.pred81.hkl')
reference = load_pickle('test.references81.pkl')
thres = 0.3
candidate = []
for instance in init_pred:
    pred = []
    for i, label in enumerate(instance):
        if label > thres:
            pred.append(i)
    if len(pred) == 0:
        pred.append(np.argmax(instance))
    candidate.append(pred)

g = open('resnet_baseline.txt', 'w')
word_to_idx = load_word_to_idx(data_path='/home/jason6582/sfyc/attention-tensorflow/nus-wide/nusdata', split='train')
refsNum = 0
cansNum = 0
correctNum = 0
classwise_num = np.zeros((3,81))
# for i in range(10):
for i in range(len(candidate)):
    refs = str(reference[i][0][:-2]).split()
    refsNum += len(refs)
    cans = candidate[i][:]
    cansNum += len(cans)
    refsDict = {}
    correct = 0
    for c in cans:
        refsDict[c] = 0
        classwise_num[0][c] += 1.0
    for r in refs:
        idx = word_to_idx[r]
        refsDict[idx-3] = 1
        classwise_num[1][idx-3] += 1.0
    for c in cans:
        if refsDict[c] == 1:
            correct += 1
            classwise_num[2][c] += 1.0
    correctNum += correct
recall = float(correctNum)/float(refsNum)
precision = float(correctNum)/float(cansNum)
o_f1 = 2.0/((1.0/recall) + (1.0/precision))
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
g.write('Average: ' + str((c_f1+o_f1)/2) + '\n')
