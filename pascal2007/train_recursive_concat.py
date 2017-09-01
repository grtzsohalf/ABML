from core.solver_recursive_concat import CaptioningSolver
from core.model_recursive_concat import CaptionGenerator
from core.utils_pascal import *
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

def main():
    word_to_idx = load_word_to_idx(data_path='./pascaldata', split='train')
    model = CaptionGenerator(word_to_idx, dim_feature=[196, 1024], dim_embed=64,
                            dim_hidden=1024, n_time_step=8, prev2out=True,
                            ctx2out=True, alpha_c=1.0, selector=True, dropout=True, batch_size=128)
    data_path = './pascaldata'
    solver = CaptioningSolver(model, data_path, n_epochs=500, batch_size=128,
                update_rule='adam', learning_rate=0.0005, print_every=30, save_every=10,
                pretrained_model=None, model_path='model/lstm/',
                test_model='model/lstm/model-1', print_bleu=True, log_path='pascallog/',
                V=len(word_to_idx), n_time_step=8)
    solver.train()

if __name__ == "__main__":
    main()
