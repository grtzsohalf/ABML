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
from tqdm import tqdm

class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()
        resnet152 = models.resnet152(pretrained=True)
        resnet152 = nn.Sequential(*list(resnet152.children())[:-3])
        for param in resnet152.parameters():
            param.requires_grad = False
        self.resnet152 = resnet152

        self.fc = nn.Sequential(
            nn.Linear(196 * 1024, 512),
            nn.Linear(512, 512),
            nn.Linear(512, 80),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.resnet152(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

resnet = Resnet()
resnet = nn.DataParallel(resnet).cuda()
resnet.load_state_dict(torch.load('model/resnet_last4_5.pth.tar'))

# Load validation data first
print 'Loading validation...'
reference = load_pickle('/home/jason6582/sfyc/attention-tensorflow/mscoco/cocodata/val/val.references.pkl')
for key, value in reference.iteritems():
    reference[key] = [int(idx) for idx in value[0].split()[:-1]]

anno_path = '/home/jason6582/sfyc/attention-tensorflow/mscoco/cocodata/val/val.annotations.pkl'
with open(anno_path, 'rb') as f:
    annotations = pickle.load(f)
image_path = list(annotations['file_name'].unique())
n_examples = len(image_path)
batch_size = 8
data_path = '/home/jason6582/sfyc/attention-tensorflow/mscoco/cocodata'
data = load_coco_data(data_path=data_path, split='val')
captions = data['captions']
groundtruth = np.zeros((n_examples, 80), dtype=np.float32)
for n, caption in enumerate(captions):
    for index in caption:
        if index > 2:
            groundtruth[n][index-3] = 1.0
all_preds = np.ndarray([n_examples, 80], dtype=np.float32)
for start, end in zip(range(0, n_examples, batch_size),
                    range(batch_size, n_examples + batch_size, batch_size)):
    image_batch_file = image_path[start:end]
    input_batch = np.array(map(lambda x: ndimage.imread(x, mode='RGB'),\
            image_batch_file))
    input_batch = input_batch.astype(np.float32)
    input_batch = np.transpose(input_batch, (0, 3, 1, 2))
    input_batch = torch.Tensor(input_batch).cuda()
    input_var = Variable(input_batch).cuda()
    groundtruth_batch = groundtruth[start:end]
    groundtruth_batch = torch.Tensor(groundtruth_batch).cuda()
    target_var = Variable(groundtruth_batch, requires_grad=False).cuda()
    pred = resnet(input_var)
    pred = np.reshape(pred.data.cpu().numpy(), [-1, 80])
    all_preds[start:end, :] = pred
candidate = []
thres = 0.3
for instance in all_preds:
    pred = []
    for i, label in enumerate(instance):
        if label > thres:
            pred.append(i)
    if len(pred) == 0:
        pred.append(np.argmax(instance))
    candidate.append(pred)
num = 0
refsNum = 0
cansNum = 0
correctNum = 0
classwise_num = np.zeros((3,80))
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
        refsDict[c] = 0  # follow idx of word_to_idx (keep 0, 1, 2)
        classwise_num[0][c] += 1.0 # idx start from 0 for convenience
    for r in refs:
        refsDict[r] = 1
        classwise_num[1][r] += 1.0
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
c_recall = np.mean(recall_arr)
c_precision = np.mean(precision_arr)
c_f1 = 2.0/((1.0/c_recall) + (1.0/c_precision))
print 'o_recall', recall
print 'o_precision', precision
print 'o_F1: ', o_f1
print 'c_recall', c_recall
print 'c_precision', c_precision
print 'c_F1: ', c_f1

