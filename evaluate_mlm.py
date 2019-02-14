from mlm_training.dataset import Dataset, ground_truth
from mlm_training.utils import safe_mkdir
from mlm_training.model import Model, make_model_from_dataset, make_save_path, make_model_param, make_train_param

import os

import numpy as np
from datetime import datetime
import sys
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('save_path',type=str,help="folder to save the checkpoints (inside ckpt folder)")
parser.add_argument('data_path',type=str,help="folder containing the split dataset")
timestep = parser.add_mutually_exclusive_group()
timestep.add_argument('-quant',action='store_true',help="use quantised timesteps")
timestep.add_argument('-event',action='store_true',help="use event timesteps")


args = parser.parse_args()

if args.quant:
    timestep_type = 'quant'
    max_len = 300
elif args.event:
    timestep_type = 'event'
    max_len = 100
else:
    timestep_type = 'time'
    max_len = 750


note_range = [21,109]
note_min = note_range[0]
note_max = note_range[1]


n_hidden = 256 #number of features in hidden layer

data = Dataset()
data.load_data_one(args.data_path,subset='test',timestep_type=timestep_type,note_range=note_range)
data.note_range = note_range

model_param = make_model_param()
model_param['n_hidden']=n_hidden
model_param['learning_rate']=0
model_param['chunks']=max_len


save_path = args.save_path
model = make_model_from_dataset(data,model_param)
model.print_params()

dataset, seq_lens = data.get_dataset_chunks_no_pad('test',max_len)

result_GT,result_s, result_th = model.compute_eval_metrics_pred(dataset,seq_lens,0.5,save_path)

print(f"XE_GT: {result_GT[0]},XE_tr_GT: {result_GT[1]},F0_GT: {result_GT[2]}")
print(f"XE_s: {result_s[0]},XE_tr_s: {result_s[1]},F0_s: {result_s[2]}")
print(f"XE_th: {result_th[0]},XE_tr_th: {result_th[1]},F0_th: {result_th[2]}")
