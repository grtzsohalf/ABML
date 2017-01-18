from core.utils import *
from prepro81 import _build_vocab
from collections import Counter
import numpy as np
import hickle

word_to_idx = {}
for part in range(16):
    annotations = load_pickle('data/train/train.annotations81_%s.pkl' % str(part))
    word_to_idx_part = _build_vocab(annotations=annotations, threshold=1)
    for key in word_to_idx_part:
        word_to_idx[key] = 0
word_list = sorted(word_to_idx.iterkeys())
print len(word_to_idx)
for i, word in enumerate(word_list):
    word_to_idx[word] = i
    print word
    print i
save_pickle(word_to_idx, 'data/train/word_to_idx81.pkl')
