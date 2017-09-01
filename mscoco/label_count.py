import numpy as np
from tqdm import tqdm
from core.utils_coco import *
word_to_idx = load_word_to_idx(data_path='./cocodata', split='train')
word2idx = load_word2idx(data_path='./cocodata', split='train')
idx_to_word = {i+3: w for w, i in word2idx.iteritems()}
idx_to_word[0] = '<NULL>'
idx_to_word[1] = '<START>'
idx_to_word[2] = '<END>'
data_path = './cocodata'
count = np.zeros(80, dtype=int)
for part in tqdm(range(20)):
    data = load_coco_data(data_path=data_path, split='train', \
                                part=str(part), load_init_pred=True)
    captions = data['captions']
    for cap in captions:
        for idx in cap:
            if idx > 2:
                count[idx-3] += 1
with open('label_count.txt', 'w') as f:
    l = []
    for i in range(len(word_to_idx)-3):
        l.append((i, count[i]))
        l = sorted(l, key=lambda tup: tup[1], reverse=True)
    for tup in l:
        f.write(str(tup[0]+3) +'\n')


