from core.solver_2_predicted import CaptioningSolver
from core.model_2_predicted import CaptionGenerator
from core.utils import *
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES']='1'

modelname = sys.argv[1]
filename = sys.argv[2]
thres = sys.argv[3]
print modelname
print filename
print thres

def main():
    word_to_idx = load_word_to_idx(data_path='./data', split='train')
    val_data = load_coco_data(data_path='./data', split='val')
    # test_data = load_coco_data(data_path='./data', split='test')
    model = CaptionGenerator(word_to_idx, dim_feature=[196, 512], dim_embed=16,
                            dim_hidden=1024, n_time_step=11, prev2out=True,
                            ctx2out=True, alpha_c=1.0, selector=True, dropout=True)
    data_path = './data'
    solver = CaptioningSolver(model, data_path, val_data, n_epochs=200, batch_size=1,
                update_rule='adam', learning_rate=0.0005, print_every=100, save_every=1,
                pretrained_model=None, model_path='model/lstm/',
                test_model=('model/lstm/%s' %modelname), print_bleu=True, log_path='log/', V=len(word_to_idx))
    solver.test(val_data, split='val', filename=filename, attention_visualization=False, thres=float(thres))
    # solver.test(test_data, split='test')
    # part=15
    # test_data = load_coco_data(data_path='./data', split='train', part=str(part))
    # solver.test(test_data, split='train', part=part)

if __name__ == "__main__":
    main()
