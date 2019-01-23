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
parser.add_argument('-quant',action='store_true',help="use quantised timesteps")
parser.add_argument('-epochs',type=int,default=1000,help="maximum number of epochs")
parser.add_argument('-lr',type=float,default=0.01,help="learning rate")
parser.add_argument('-use_focal_loss',action='store_true',help="use focal loss instead of usual cross-entropy loss")

args = parser.parse_args()

if args.quant:
    fs = 4
    max_len = 300
else:
    fs = 25
    max_len = 750

note_range = [21,109]
note_min = note_range[0]
note_max = note_range[1]


n_hidden = 256 #number of features in hidden layer
learning_rate = args.lr


train_param = make_train_param()
train_param['epochs']=args.epochs
train_param['batch_size']=5
train_param['display_per_epoch']=None
train_param['save_step']=1
train_param['max_to_keep']=1
train_param['summarize']=True
train_param['early_stop_epochs']=100

print("Computation start : "+str(datetime.now()))

data = Dataset(rand_transp=True)
data.load_data(args.data_path,note_range=note_range,
    fs=fs,quant=args.quant)
# data.transpose_all()


model_param = make_model_param()
model_param['n_hidden']=n_hidden
model_param['learning_rate']=learning_rate
model_param['chunks']=max_len
model_param['use_focal_loss']=args.use_focal_loss


save_path = args.save_path
log_path = os.path.join("ckpt",save_path)
safe_mkdir(log_path)
# f= open(os.path.join(log_path,"log.txt"), 'w')
# sys.stdout = f


model = make_model_from_dataset(data,model_param)
model.print_params()
model.train(data,save_path=save_path,train_param=train_param)



print("Computation end : "+str(datetime.now()))
