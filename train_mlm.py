from mlm_training.dataset import Dataset, ground_truth
from datasetMaps import DatasetMaps
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
parser.add_argument('-n_hidden',type=int,default=256,help="number of hidden nodes (default=256)")
parser.add_argument('-epochs',type=int,default=1000,help="maximum number of epochs")
parser.add_argument('-early_stop_epochs',type=int,default=100,help="stop training after this number of epochs without improvement on valid set")
parser.add_argument('-lr',type=float,default=0.01,help="learning rate")
parser.add_argument('-use_focal_loss',action='store_true',help="use focal loss instead of usual cross-entropy loss")
parser.add_argument('-resume',action='store_true',help="resume training from latest checkpoint in save_path")
parser.add_argument('-sched_sampl',type=str,help="type of sampling for scheduled sampling. If not specified, no scheduled sampling")
parser.add_argument('-sched_sampl_load',type=str,help="path of the pretrained model to load. If not specified, the model will be trained from scratch")
parser.add_argument('-sampl_mix_weight',type=float,help="mix weight when sampling from a mix of acoustic and language priors")
parser.add_argument('-sched_shape',type=str,default="linear",help="shape of the sampling likelihood curve")
parser.add_argument('-sched_dur',type=int,default=1000,help="duration in epochs of the schedule (if lower than epochs, sampling will always be applied after the end of schedule)")
parser.add_argument('-pitchwise',type=int,help='to train a pitch-wise model; value gives width of pitch window.')


args = parser.parse_args()

if args.sched_sampl == 'mix' and (args.sched_sampl_load is None or args.sampl_mix_weight is None):
    raise ValueError('when using -sched_sampl mix , -sched_sampl_load and -sampl_mix_weight have to be provided!')


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


n_hidden = args.n_hidden #number of features in hidden layer
learning_rate = args.lr


train_param = make_train_param()
train_param['epochs']=args.epochs
train_param['batch_size']=5
train_param['display_per_epoch']=None
train_param['save_step']=1
train_param['max_to_keep']=1
train_param['summarize']=True
train_param['early_stop_epochs']=args.early_stop_epochs
train_param['schedule_shape']=args.sched_shape
train_param['schedule_duration']=args.sched_dur



print("Computation start : "+str(datetime.now()))

if args.sched_sampl == 'mix':
    data= DatasetMaps()
    data.load_data(args.data_path,timestep_type=timestep_type,note_range=note_range,subsets=['train','valid'])
else:
    data = Dataset(rand_transp=True)
    data.load_data(args.data_path,timestep_type=timestep_type,note_range=note_range)
# data.transpose_all()


model_param = make_model_param()
model_param['n_hidden']=n_hidden
model_param['learning_rate']=learning_rate
model_param['chunks']=max_len
model_param['use_focal_loss']=args.use_focal_loss
model_param['scheduled_sampling']=args.sched_sampl
if args.pitchwise is None:
    model_param['pitchwise']=False
else:
    model_param['pitchwise']=True
    model_param['n_notes'] = 2*args.pitchwise+1
    train_param['batch_size']=500



save_path = args.save_path
log_path = os.path.join("ckpt",save_path)
safe_mkdir(log_path)
# f= open(os.path.join(log_path,"log.txt"), 'w')
# sys.stdout = f


model = make_model_from_dataset(data,model_param)
model.print_params()

if args.resume:
    safe_mkdir(os.path.join(save_path,'resume'))
    model.resume_training(save_path,data,os.path.join(save_path,'resume'),train_param)
else:
    if args.sched_sampl_load is not None:
        model.resume_training(args.sched_sampl_load,data,save_path,train_param)
    else:
        model.train(data,save_path=save_path,train_param=train_param)



print("Computation end : "+str(datetime.now()))
