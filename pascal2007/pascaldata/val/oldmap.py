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
for i in range(len(candidate)):
    for j in range(len(candidate[i])):
        candidate[i][j] = 3 + int(candidate[i][j])
g = open(resultFile, 'w')

word_to_idx = load_word_to_idx(data_path='/home/jason6582/sfyc/attention-tensorflow/pascaldata',\
                            split='train')
# print word_to_idx
idx_to_word = {i:w for w, i in word_to_idx.iteritems()}

classwise_ap =[ [] for i in range(23)]
mAP = []
for i in range(len(candidate)):
    refs = reference[i]
    cans = candidate[i]
    ref_list = [0] * 20
    instance_ap = []
    for j, can in enumerate(cans):
        if can in refs:
            ref_list[j] = 1
    print ref_list
    order = {}
    temp_ap = {}
    for ref in refs:
        pos = cans.index(ref)
        order[pos] = ref
        temp_ap[pos] = float(sum(ref_list[:pos+1])) / float(pos+1)
    temp_ap = list(reversed([temp_ap[pos] for pos in sorted(order.keys())]))
    print temp_ap
    ap_sum = 0.0
    for j in range(11):
        recall = float(j/10)
        max = 0.0
        for k, pos in enumerate(list(reversed(sorted(order.keys())))):
            ref 
            if float(sum(ref_list[pos:])) / float(sum(ref_list)) > recall and 
    '''
    ap_sum = 0.0
    for j, pos in enumerate(list(reversed(sorted(order.keys())))):
        ref = order[pos]
        classwise_ap[ref].append(max(temp_ap[:j+1]))
        temp_ap[j] = max(temp_ap[:j+1])
    print temp_ap
    mAP.append(sum(temp_ap)/float(len(temp_ap)))
    '''
mAP = sum(mAP)/len(mAP)
classwise_ap = classwise_ap[3:]
for i in range(len(classwise_ap)):
    classwise_ap[i] = sum(classwise_ap[i])/float(len(classwise_ap[i]))
for i, key in enumerate(sorted(word_to_idx.keys())[3:]):
    print key + "     " +  str(classwise_ap[i])
print mAP
print sum(classwise_ap) / 20.0
