from core.solver_recursive_concat import CaptioningSolver
from core.model_recursive_concat import CaptionGenerator
from core.utils_nus import *
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES']='1'

modelname = sys.argv[1]
filename = sys.argv[2]
thres = sys.argv[3]
print '#########################'
print 'model = ' + modelname
print 'thres = ' + thres
print '#########################'

def main():
    word_to_idx = load_word_to_idx(data_path='./nusdata', split='train')
    model = CaptionGenerator(word_to_idx, dim_feature=[196, 1024], dim_embed=64,
                            dim_hidden=1024, n_time_step=11, prev2out=True,
                            ctx2out=True, alpha_c=1.0, selector=True, dropout=True)
    data_path = './nusdata'
    solver = CaptioningSolver(model, data_path, n_epochs=200, batch_size=1,
                update_rule='adam', learning_rate=0.0005, print_every=100, save_every=1,
                pretrained_model=None, model_path='model/',
                test_model=('model/%s' %modelname), print_bleu=True, log_path='log/', 
                V=len(word_to_idx))
    solver.test(split='val', filename=filename, attention_visualization=False, \
                thres=float(thres))

if __name__ == "__main__":
    main()
