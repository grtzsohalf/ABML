import sys
sys.path.append('/home/jason6582/sfyc/attention-tensorflow/mscoco')

from scipy import ndimage
from torch.autograd import Variable
from core.utils_coco import *
import os
import time
import numpy as np
import cPickle as pickle
import hickle
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torchvision.models as models
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

resnet152 = models.resnet152(pretrained=True)
layers = []
layers.append(nn.Linear(2048, 1024))
layers.append(nn.Linear(1024, 1024))
layers.append(nn.Linear(1024, 80))
layers.append(nn.Sigmoid())
resnet152.fc = nn.Sequential(*layers)
resnet152 = nn.DataParallel(resnet152).cuda()
model_name = 'adam_3fc_lr0.0015'
model_path = 'model/resnet_init_pred_%s.pth.tar' % model_name
resnet152.load_state_dict(torch.load(model_path))
part_num = 20
batch_size = 16

criterion = nn.BCELoss().cuda()
split = 'val'
anno_path = '/home/jason6582/sfyc/attention-tensorflow/mscoco/cocodata/%s/%s.annotations.pkl'\
            % (split, split)
with open(anno_path, 'rb') as f:
    annotations = pickle.load(f)
image_path = list(annotations['file_name'].unique())
n_examples = len(image_path)

data_path = '/home/jason6582/sfyc/attention-tensorflow/mscoco/cocodata'
data = load_coco_data(data_path=data_path, split=split)
captions = data['captions']
groundtruth = np.zeros((n_examples, 80), dtype=np.float32)
for n, caption in enumerate(captions):
    for index in caption:
        if index > 2:
            groundtruth[n][index-3] = 1.0

# all_feats = np.ndarray([n_examples, 196, 1024], dtype=np.float32)
init_pred = []
all_loss = 0
for start, end in zip(range(0, n_examples, batch_size),
                    range(batch_size, n_examples + batch_size, batch_size)):
    image_batch_file = image_path[start:end]
    input_batch = np.array(map(lambda x: ndimage.imread(x, mode='RGB'),\
            image_batch_file))
    input_batch = input_batch.astype(np.float32)
    input_batch = np.transpose(input_batch, (0, 3, 1, 2))
    input_batch = torch.Tensor(input_batch).cuda()
    input_var = Variable(input_batch, volatile=True).cuda()
    groundtruth_batch = groundtruth[start:end]
    groundtruth_batch = torch.Tensor(groundtruth_batch).cuda()
    target_var = Variable(groundtruth_batch, volatile=True).cuda()

    output = resnet152(input_var)
    init_pred += output.data.cpu().numpy().tolist()
    '''
    if start == 0:
        print 'groundtruth, ', target_var.data.cpu().numpy()[:3]
        print 'output, ', output.data.cpu().numpy()[:3]
        print 'diff, ', output.data.cpu().numpy()[:3] - target_var.data.cpu().numpy()[:3]
    '''
    loss = criterion(output, target_var)
    all_loss += loss.data.cpu().numpy()[0]
print "Loss of model = ", all_loss

reference = load_pickle('/home/jason6582/sfyc/attention-tensorflow/mscoco/cocodata/val/val.references.pkl')
for key, value in reference.iteritems():
    reference[key] = [int(idx) for idx in value[0].split()[:-1]]

thres = 0.2
candidate = []
for instance in init_pred:
    pred = []
    for i, label in enumerate(instance):
        if label > thres:
            pred.append(i)
    candidate.append(pred)

g = open('result_init_pred_%s.txt' % model_name, 'w')
word_to_idx = load_word2idx(data_path='/home/jason6582/sfyc/attention-tensorflow/mscoco/cocodata', split='train')
idx_to_word = {i:w for w, i in word_to_idx.iteritems()}

refsNum = 0
cansNum = 0
correctNum = 0
classwise_num = np.zeros((3,80))
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
        can_str.append(idx_to_word[c])
    for r in refs:
        ref_str.append(idx_to_word[r])
    for c in cans:
        refsDict[c] = 0  # follow idx of word_to_idx (keep 0, 1, 2)
        classwise_num[0][c] += 1.0 # idx start from 0 for convenience
    for r in refs:
        refsDict[r] = 1
        # idx = word_to_idx[r]
        classwise_num[1][r] += 1.0
    for c in cans:
        if refsDict[c] == 1:
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

