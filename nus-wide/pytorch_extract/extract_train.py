import sys
from scipy import ndimage
from torch.autograd import Variable
import os
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
resnet152 = nn.Sequential(*list(resnet152.children())[:-3])
resnet152 = nn.DataParallel(resnet152).cuda()

part_num = 20
batch_size = 128

split = 'train'
for part in range(part_num):
    print "part", part, "of %s features" % split
    anno_path = '/home/jason6582/sfyc/attention-tensorflow/nus-wide/nusdata/%s/%s.annotations81_%s.pkl'\
                % (split, split, str(part))
    save_path = '/home/jason6582/sfyc/attention-tensorflow/nus-wide/nusdata/%s/%s.features81_%s.hkl'\
                % (split, split, str(part))
    with open(anno_path, 'rb') as f:
        annotations = pickle.load(f)
    image_path = list(annotations['file_name'].unique())
    n_examples = len(image_path)

    all_feats = np.ndarray([n_examples, 196, 1024], dtype=np.float32)
    for start, end in zip(range(0, n_examples, batch_size),
                        range(batch_size, n_examples + batch_size, batch_size)):
        image_batch_file = image_path[start:end]
        image_batch = np.array(map(lambda x: ndimage.imread(x, mode='RGB'),\
                image_batch_file))
        image_batch = image_batch.astype(np.float32)
        image_batch = np.transpose(image_batch, (0, 3, 1, 2))
        image_batch = torch.Tensor(image_batch).cuda()
        image_var = Variable(image_batch, volatile=True).cuda()
        feats = resnet152(image_var)
        feats = np.reshape(feats.data.cpu().numpy(), [-1, 1024, 196])
        feats = np.transpose(feats, (0, 2, 1))
        all_feats[start:end, :] = feats
        print ("Processed %d %s features.." % (end, split))
    # use hickle to save huge feature vectors
    hickle.dump(all_feats, save_path)
    print ("Saved %s.." % (save_path))
'''
saved_part = 0
for i, (input, target) in enumerate(loader):
    if i % part_batch == 0:
        if i == 0:
            part_feature = np.ndarray([part_examples, 196, 1024], dtype=np.float32)
        else:
            save_path = './coco_train_feature_%d.hkl' % saved_part
            hickle.dump(part_feature, save_path)
            print 'part %d feature saved.' % (saved_part+1)
            saved_part += 1
            if saved_part != part_num-1:
                part_feature = np.ndarray([part_examples, 196, 1024], dtype=np.float32)
            else:
                part_feature = np.ndarray([last_part_examples, 196, 1024], dtype=np.float32)
    start = i * batch_size - saved_part*part_examples
    end = (i+1) * batch_size - saved_part*part_examples
    input_var = Variable(input, volatile=True)
    feature = resnet50(input_var)
    numpy_feature = np.reshape(feature.data.numpy(), [-1, 1024, 196])
    numpy_feature = np.transpose(numpy_feature, (0, 2, 1))
    part_feature[start:end, :] = numpy_feature
    print ("Processed %d features of part %d..." % (end, (saved_part+1)))
save_path = './coco_train_feature_%d.hkl' % saved_part
hickle.dump(part_feature, save_path)
print 'part %d feature saved.' % (saved_part+1)
'''

