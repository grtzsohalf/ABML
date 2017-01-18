from core.solver_2 import CaptioningSolver
from core.model_2 import CaptionGenerator
from core.utils import *
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

def main():
    word_to_idx = load_word_to_idx(data_path='./data', split='train')
    val_data = load_coco_data(data_path='./data', split='val')
    model = CaptionGenerator(word_to_idx, dim_feature=[196, 512], dim_embed=len(word_to_idx),
                            dim_hidden=1024, n_time_step=21, prev2out=True,
                            ctx2out=True, alpha_c=1.0, selector=True, dropout=True)
    data_path = './data'
    solver = CaptioningSolver(model, data_path, val_data, n_epochs=500, batch_size=128,
                update_rule='adam', learning_rate=0.001, print_every=100, save_every=1,
                pretrained_model=None, model_path='model/lstm/',
                test_model='model/lstm/model-100', print_bleu=True, log_path='log/', V=len(word_to_idx))
    solver.test(val_data, split='val')

if __name__ == "__main__":
    main()
