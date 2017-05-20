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
layers.append(nn.Linear(2048, 80))
layers.append(nn.Sigmoid())
resnet152.fc = nn.Sequential(*layers)
resnet152 = nn.DataParallel(resnet152).cuda()
model_path = 'model/resnet_init_pred_6.pth.tar'
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

all_feats = np.ndarray([n_examples, 196, 1024], dtype=np.float32)
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

    if start == 0:
        print 'groundtruth, ', target_var.data.cpu().numpy()[:3]
        print 'output, ', output.data.cpu().numpy()[:3]
        print 'diff, ', output.data.cpu().numpy()[:3] - target_var.data.cpu().numpy()[:3]

    loss = criterion(output, target_var)
    all_loss += loss.data.cpu().numpy()[0]
print "Loss of model = ", all_loss

