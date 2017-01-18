import cPickle as pickle

with open('./data/val/val.candidate.captions.pkl', 'r') as f:
    captions = pickle.load(f)

print captions
