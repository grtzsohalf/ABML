import sys
sys.path.append('/home/jason6582/sfyc/attention-tensorflow/pascal2007')

import tensorflow as tf
import matplotlib.pyplot as plt
import skimage.transform
import numpy as np
import time
import math
import os
import cPickle as pickle
from scipy import ndimage
from core.utils_pascal import *

split = 'test'
model_type = 'iterative_update'
resultFile = 'iterative_update-290_2.txt'
e_list = ['290', '300', '310', '330','340']
epoch_num = 5
predict = np.zeros((4952, 20), dtype=np.float32)
for i in range(epoch_num):
    e = e_list[i]
    p = load_pickle('/home/jason6582/sfyc/attention-tensorflow/pascal2007/pascaldata/%s/%s-%s_pred.pkl'\
                                % (split, model_type, e))
    predict += p

predict = np.transpose(predict, (1, 0))
loaded_reference = load_pickle('/home/jason6582/sfyc/attention-tensorflow/pascal2007/pascaldata/%s/%s.references.pkl'\
                                % (split, split))
reference = np.zeros(predict.shape)
for key, value in loaded_reference.iteritems():
    answer = []
    for label in value[0]:
        reference[label-3][key] = 1
g = open(resultFile, 'w')
word_to_idx = load_word_to_idx(data_path='/home/jason6582/sfyc/attention-tensorflow/pascal2007/pascaldata',\
                split='train')
idx_to_word = {i:w for w, i in word_to_idx.iteritems()}
label_list = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', \
                'chair', 'cow', 'dining_table', 'dog', 'horse', 'motorbike', 'person', \
                'plant', 'sheep', 'sofa', 'train', 'tv']
all_map = []
for i, label in enumerate(label_list):
    map_dic = {}
    for j, instance in enumerate(reference[i]):
        map_dic[predict[i][j]] = reference[i][j]
    map_list = []
    total_count = 0.0
    correct_count = 0.0
    for key in reversed(sorted(map_dic.iterkeys())):
        total_count += 1.0
        if map_dic[key] == 1:
            correct_count += 1.0
            map_list.append(correct_count/total_count)
    l = len(map_list)
    for i in range(l-1):
        if map_list[l-i-1] > map_list[l-i-2]:
            map_list[l-i-2] = map_list[l-i-1]
    all_map.append(sum(map_list)/len(map_list))
    g.write(label + '\nmAP: ' + str(sum(map_list)/len(map_list)) + '\n\n')
g.write('Average: ' + str(sum(all_map)/len(all_map)))
