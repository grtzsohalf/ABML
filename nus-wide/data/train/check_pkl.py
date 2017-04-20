import cPickle as pickle

with open('train.captions81_1.pkl', 'r') as f:
    captions = pickle.load(f)
    print captions[:10]
