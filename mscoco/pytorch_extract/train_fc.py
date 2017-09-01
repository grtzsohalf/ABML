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
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

resnet152 = models.resnet152(pretrained=True)
for param in resnet152.parameters():
    param.requires_grad = False
layers = []
layers.append(nn.Linear(2048, 80))
# layers.append(nn.Linear(512, 512))
# layers.append(nn.Linear(512, 80))
layers.append(nn.Sigmoid())
resnet152.fc = nn.Sequential(*layers)
training_epoch = 3
save_every = 1
test_every = 1
part_num = 20
batch_size = 8

criterion = nn.BCELoss().cuda()
optimizer = torch.optim.SGD(resnet152.fc.parameters(),
                            0.1, # Learning rate
                            momentum=0.9,
                            weight_decay=1e-4)
resnet152 = nn.DataParallel(resnet152).cuda()

def evaluate(thres, all_preds, reference, save_file):
    candidate = []
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
    with open(save_file, 'a') as f:
        f.write('thres ='+str(thres)+'\n')
        f.write('overall F1: '+str(o_f1)+'\n')
        f.write('classwise F1: '+str(c_f1)+'\n')

# Training part
print 'Training starts...'
split = 'train'
prev_loss = []
curr_loss = []
for i in range(part_num):
    prev_loss.append(-1)
    curr_loss.append(0)
start_time = time.time()
for epoch in range(training_epoch):
    print "Epoch %d" % (epoch+1)
    arr = np.arange(20)
    np.random.shuffle(arr)
    epoch_loss = 0
    for p in tqdm(range(part_num)):
        part = arr[p]
        part_loss = 0
        anno_path = '/home/jason6582/sfyc/attention-tensorflow/mscoco/cocodata/%s/%s.annotations_%s.pkl'\
                    % (split, split, str(part))
        with open(anno_path, 'rb') as f:
            annotations = pickle.load(f)
        f.close()
        image_path = list(annotations['file_name'].unique())
        n_examples = len(image_path)
        data_path = '/home/jason6582/sfyc/attention-tensorflow/mscoco/cocodata'
        data = load_coco_data(data_path=data_path, split='train', part=str(part))
        captions = data['captions']
        groundtruth = np.zeros((n_examples, 80), dtype=np.float32)
        for n, caption in enumerate(captions):
            for index in caption:
                if index > 2:
                    groundtruth[n][index-3] = 1.0
        for start, end in zip(range(0, n_examples, batch_size),
                            range(batch_size, n_examples + batch_size, batch_size)):
            image_batch_file = image_path[start:end]
            input_batch = np.array(map(lambda x: ndimage.imread(x, mode='RGB'),\
                    image_batch_file))
            input_batch = input_batch.astype(np.float32)
            input_batch = np.transpose(input_batch, (0, 3, 1, 2))
            input_batch = torch.Tensor(input_batch)
            input_var = Variable(input_batch).cuda()
            groundtruth_batch = groundtruth[start:end]
            groundtruth_batch = torch.Tensor(groundtruth_batch)
            target_var = Variable(groundtruth_batch, requires_grad=False).cuda()

            output = resnet152(input_var)
            loss = criterion(output, target_var)
            part_loss += loss.data.cpu().numpy()[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            del input_var
            del target_var
            del loss
        epoch_loss += part_loss
    print "Epoch %s loss: " % str(epoch+1), epoch_loss
    epoch_loss = 0
    print "Elapsed time: ", time.time() - start_time
    
    if (epoch+1) % test_every == 0:
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
        data_path = '/home/jason6582/sfyc/attention-tensorflow/mscoco/cocodata'
        data = load_coco_data(data_path=data_path, split='val')
        captions = data['captions']
        all_preds = np.ndarray([n_examples, 80], dtype=np.float32)
        for start, end in zip(range(0, n_examples, batch_size),
                            range(batch_size, n_examples + batch_size, batch_size)):
            image_batch_file = image_path[start:end]
            input_batch = np.array(map(lambda x: ndimage.imread(x, mode='RGB'),\
                    image_batch_file))
            input_batch = input_batch.astype(np.float32)
            input_batch = np.transpose(input_batch, (0, 3, 1, 2))
            input_batch = torch.Tensor(input_batch).cuda()
            input_var = Variable(input_batch, volatile=True).cuda()
            pred = resnet152(input_var)
            pred = np.reshape(pred.data.cpu().numpy(), [-1, 80])
            all_preds[start:end, :] = pred
        for thres in range(1, 5, 1):
            evaluate(thres*0.1, all_preds, reference, 'train_accuracy.txt')

    if (epoch+1) % save_every == 0:
        filename = 'model/resnet_fc_1layer_%s.pth.tar' % str(epoch+1)
        print filename, 'saved.\n'
        torch.save(resnet152.state_dict(), filename)

