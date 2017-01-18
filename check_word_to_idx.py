from core.utils import *

word_to_idx = load_word_to_idx(data_path='./data', split='train')
print len(word_to_idx)
for i in sorted(word_to_idx.iteritems()):
    print i
