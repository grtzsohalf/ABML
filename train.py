from core.solver_nus import CaptioningSolver
from core.model_nus import CaptionGenerator
from core.utils import *
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

def main():
    word_to_idx = load_word_to_idx(data_path='./data', split='train')
    model = CaptionGenerator(word_to_idx, dim_feature=[196, 512], dim_embed=16,
                            dim_hidden=1024, n_time_step=11, prev2out=True,
                            ctx2out=True, alpha_c=1.0, selector=True, dropout=True, batch_size=128)
    data_path = './data'
    solver = CaptioningSolver(model, data_path, n_epochs=50, batch_size=128,
                update_rule='adam', learning_rate=0.0003, print_every=30, save_every=1,
                pretrained_model=None, model_path='model/lstm/',
                test_model='model/lstm/model-1', print_bleu=True, log_path='log/', V=len(word_to_idx), n_time_step=11)
    solver.train()

if __name__ == "__main__":
    main()
