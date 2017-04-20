from core.solver_coco import CaptioningSolver
from core.model_coco import CaptionGenerator
from core.utils_coco import *
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

def main():
    word_to_idx = load_word_to_idx(data_path='./cocodata', split='train')
    word2idx = load_word2idx(data_path='./cocodata', split='train')
    idx_to_word = {i+3: w for w, i in word2idx.iteritems()}
    idx_to_word[0] = '<NULL>'
    idx_to_word[1] = '<START>'
    idx_to_word[2] = '<END>'
    model = CaptionGenerator(word_to_idx, idx_to_word, dim_feature=[196, 512], dim_embed=16,
                            dim_hidden=1024, n_time_step=16, prev2out=True,
                            ctx2out=True, alpha_c=1.0, selector=True, dropout=True, batch_size=128)
    data_path = './cocodata'
    solver = CaptioningSolver(model, data_path, n_epochs=100, batch_size=128,
                update_rule='adam', learning_rate=0.0003, print_every=100, save_every=1,
                pretrained_model='model/lstm/mscoco_old-13', model_path='model/lstm/',
                test_model='model/lstm/model-1', print_bleu=True, log_path='cocolog/',
                V=len(word_to_idx), n_time_step=16)
    solver.train()

if __name__ == "__main__":
    main()
