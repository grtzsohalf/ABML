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
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

resnet152 = models.resnet152(pretrained=True)
# layers = list(resnet152.children())
# for param in resnet152.parameters():
#     param.requires_grad = False
layers = []
layers.append(nn.Linear(2048, 80))
layers.append(nn.Sigmoid())
resnet152.fc = nn.Sequential(*layers)
# resnet152.fc = nn.Linear(2048, 80)
resnet152 = nn.DataParallel(resnet152).cuda()
training_epoch = 50
save_every = 1
part_num = 20
batch_size = 16

criterion = nn.BCELoss().cuda()
optimizer = torch.optim.SGD(resnet152.parameters(),
                            0.1, # Learning rate
                            momentum=0.9,
                            weight_decay=1e-4)

# Training part
split = 'train'
prev_loss = []
curr_loss = []
for i in range(part_num):
    prev_loss.append(-1)
    curr_loss.append(0)
start_time = time.time()
for epoch in range(training_epoch):
    arr = np.arange(20)
    np.random.shuffle(arr)
    for p in range(part_num):
        part = arr[p]
        print "part", part, "of epoch %d" % (epoch+1)
        anno_path = '/home/jason6582/sfyc/attention-tensorflow/mscoco/cocodata/%s/%s.annotations_%s.pkl'\
                    % (split, split, str(part))
        save_path = '/home/jason6582/sfyc/attention-tensorflow/mscoco/cocodata/%s/%s.features_%s.hkl'\
                    % (split, split, str(part))
        with open(anno_path, 'rb') as f:
            annotations = pickle.load(f)
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
            target_var = Variable(groundtruth_batch).cuda()

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
        filename = 'model/resnet_epoch_%s.pth.tar' % str(epoch+1)
        print filename, 'saved.\n' 
        torch.save(resnet152.state_dict(), filename)



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
