import sys
sys.path.append('/home/jason6582/sfyc/attention-tensorflow/nus-wide')

from scipy import ndimage
from torch.autograd import Variable
from core.utils_nus import *
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
for param in resnet152.parameters():
    param.requires_grad = False
layers = []
layers.append(nn.Linear(2048, 81))
layers.append(nn.Sigmoid())
resnet152.fc = nn.Sequential(*layers)
training_epoch = 50
save_every = 1
part_num = 50
batch_size = 16

criterion = nn.BCELoss().cuda()
optimizer = torch.optim.SGD(resnet152.fc.parameters(),
                            0.1, # Learning rate
                            momentum=0.9,
                            weight_decay=1e-4)
'''
optimizer = torch.optim.Adam(resnet152.fc.parameters(),
                            0.001 # Learning rate
                            )
'''
resnet152 = nn.DataParallel(resnet152).cuda()
# Training part
split = 'train'
prev_loss = []
curr_loss = []
for i in range(part_num):
    prev_loss.append(-1)
    curr_loss.append(0)
start_time = time.time()
for epoch in range(training_epoch):
    arr = np.arange(part_num)
    np.random.shuffle(arr)
    for p in range(part_num):
        part = arr[p]
        print "part", part, "of epoch %d" % (epoch+1)
        anno_path = '/home/jason6582/sfyc/attention-tensorflow/nus-wide/nusdata/%s/%s.annotations81_%s.pkl'\
                    % (split, split, str(part))
        with open(anno_path, 'rb') as f:
            annotations = pickle.load(f)
        image_path = list(annotations['file_name'].unique())
        n_examples = len(image_path)

        data_path = '/home/jason6582/sfyc/attention-tensorflow/nus-wide/nusdata'
        data = load_nus_data(data_path=data_path, split='train', part=str(part))
        captions = data['captions']
        groundtruth = np.zeros((n_examples, 81), dtype=np.float32)
        for n, caption in enumerate(captions):
            for index in caption:
                if index > 2:
                    groundtruth[n][index-3] = 1.0

        all_feats = np.ndarray([n_examples, 196, 1024], dtype=np.float32)

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

            output = resnet152(input_var)
            loss = criterion(output, target_var)
            curr_loss[part] += loss.data.cpu().numpy()[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print "Previous epoch loss (part %s): " % str(part+1), prev_loss[part]
        print "Current epoch loss (part %s): " % str(part+1), curr_loss[part]
        print "Elapsed time: ", time.time() - start_time
        prev_loss[part] = curr_loss[part]
        curr_loss[part] = 0
    if (epoch+1) % save_every == 0:
        filename = 'model/resnet_init_pred_3fc_lr0.001%s.pth.tar' % str(epoch+1)
        print filename, 'saved.\n'
        torch.save(resnet152.state_dict(), filename)

