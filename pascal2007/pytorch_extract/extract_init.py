import sys
from scipy import ndimage
from torch.autograd import Variable
import os
import numpy as np
import cPickle as pickle
import hickle
import json
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
layers = []
layers.append(nn.Linear(2048, 20))
layers.append(nn.Sigmoid())
resnet152.fc = nn.Sequential(*layers)
resnet152 = nn.DataParallel(resnet152).cuda()
resnet152.load_state_dict(torch.load('model/resnet_init_pred_30.pth.tar'))

batch_size = 128

for split in ['train', 'val', 'test']:
    anno_path = '/home/jason6582/sfyc/attention-tensorflow/pascal2007/pascaldata/%s/%s.annotations.pkl'\
                % (split, split)
    save_path = '/home/jason6582/sfyc/attention-tensorflow/pascal2007/pascaldata/%s/%s.init.pred.hkl'\
                % (split, split)
    with open(anno_path, 'rb') as f:
        annotations = pickle.load(f)
    image_path = list(annotations['file_name'].unique())
    n_examples = len(image_path)

    all_feats = np.ndarray([n_examples, 20], dtype=np.float32)
    for start, end in zip(range(0, n_examples, batch_size),
                            range(batch_size, n_examples + batch_size, batch_size)):
        image_batch_file = image_path[start:end]
        image_batch = np.array(map(lambda x: ndimage.imread(x, mode='RGB'),\
                image_batch_file)).astype(np.float32)
        image_batch = np.transpose(image_batch, (0, 3, 1, 2))
        image_batch = torch.Tensor(image_batch).cuda()
        image_var = Variable(image_batch, volatile=True).cuda()
        feats = resnet152(image_var)
        feats = np.reshape(feats.data.cpu().numpy(), [-1, 20])
        all_feats[start:end, :] = feats
        print ("Processed %d %s features.." % (end, split))
        # use hickle to save huge feature vectors
    hickle.dump(all_feats, save_path)
    print ("Saved %s.." % (save_path))
