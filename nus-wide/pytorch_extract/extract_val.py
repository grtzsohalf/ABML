import sys
sys.path.append('/home/jason6582/sfyc/coco-api/PythonAPI')

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
resnet152 = nn.Sequential(*list(resnet152.children())[:-1])
resnet152 = nn.DataParallel(resnet152).cuda()

batch_size = 128
# loader = torch.utils.data.DataLoader(cap, batch_size=batch_size, shuffle=False)
for split in ['val', 'test']:
    anno_path = '/home/jason6582/sfyc/attention-tensorflow/nus-wide/nusdata/%s/%s.annotations81.pkl'\
                % (split, split)
    save_path = '/home/jason6582/sfyc/attention-tensorflow/nus-wide/feature_2048/%s/%s.features81.hkl'\
                % (split, split)
    with open(anno_path, 'rb') as f:
        annotations = pickle.load(f)
    image_path = list(annotations['file_name'].unique())
    n_examples = len(image_path)

    all_feats = np.ndarray([n_examples, 2048], dtype=np.float32)
    for start, end in zip(range(0, n_examples, batch_size),
                            range(batch_size, n_examples + batch_size, batch_size)):
        image_batch_file = image_path[start:end]
        image_batch = np.array(map(lambda x: ndimage.imread(x, mode='RGB'),\
                image_batch_file)).astype(np.float32)
        image_batch = np.transpose(image_batch, (0, 3, 1, 2))
        image_batch = torch.Tensor(image_batch).cuda()
        image_var = Variable(image_batch, volatile=True).cuda()
        feats = resnet152(image_var)
        feats = np.reshape(feats.data.cpu().numpy(), [-1, 2048])
        # feats = np.transpose(feats, (0, 2, 1))
        all_feats[start:end, :] = feats
        print ("Processed %d %s features.." % (end, split))
        # use hickle to save huge feature vectors
    hickle.dump(all_feats, save_path)
    print ("Saved %s.." % (save_path))

'''
for i, image in enumerate(data['images']):
    print image['file_name']
    start = i * batch_size
    end = (i+1) * batch_size
    input_var = Variable(input, volatile=True)
    feature = resnet50(input_var)
    numpy_feature = np.reshape(feature.data.numpy(), [-1, 1024, 196])
    numpy_feature = np.transpose(numpy_feature, (0, 2, 1))
    all_feature[start:end, :] = numpy_feature
    print ("Processed %d features..." % end)
save_path = './coco_val_feature.hkl'
# use hickle to save huge feature vectors
hickle.dump(all_feature, save_path)
'''
