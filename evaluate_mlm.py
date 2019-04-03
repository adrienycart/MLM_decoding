from mlm_training.dataset import Dataset, ground_truth
from datasetMaps import DatasetMaps
from mlm_training.utils import safe_mkdir
from mlm_training.model import Model, make_model_from_dataset, make_save_path, make_model_param, make_train_param
import matplotlib.pyplot as plt

import os
import pickle

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
parser.add_argument('-compare',type=str,nargs='*',help="compare with following models (can have several values)")
parser.add_argument('-sched_mix',action='store_true',help='evaluate by sampling from acoustic outputs')

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

def make_save_names(save_path):
    if args.sched_mix:
        cross_path = os.path.join('ckpt',save_path,'result_cross_mix.txt')
        cross_tr_path = os.path.join('ckpt',save_path,'result_cross_tr_mix.txt')
        F_path = os.path.join('ckpt',save_path,'result_F_mix.txt')
    else:
        cross_path = os.path.join('ckpt',save_path,'result_cross.txt')
        cross_tr_path = os.path.join('ckpt',save_path,'result_cross_tr.txt')
        F_path = os.path.join('ckpt',save_path,'result_F.txt')
    return [cross_path,cross_tr_path,F_path]


all_save_names = sum([make_save_names(path) for path in [args.save_path]+args.compare],[])
if not all([os.path.isfile(path) for path in all_save_names]):
    #If not all data has been compute already
    if args.sched_mix:
        data= DatasetMaps()
        data.load_data(args.data_path,timestep_type=timestep_type,subsets=['test'],acoustic_model='bittner')

    else:
        data = Dataset()
        data.load_data_one(args.data_path,subset='test',timestep_type=timestep_type,note_range=note_range)
    data.note_range = note_range

model_param = make_model_param()
model_param['n_hidden']=n_hidden
model_param['learning_rate']=0
model_param['chunks']=max_len
if args.sched_mix:
    model_param['scheduled_sampling'] = "mix"
    model_param['sampl_mix_weight'] = 1.0
else:
    model_param['scheduled_sampling'] = "self"


save_path = args.save_path
model = make_model_from_dataset(data,model_param)
model.print_params()

data, target, seq_lens = data.get_dataset_chunks_no_pad('test',max_len)

save_path_cross,save_path_cross_tr,save_path_f = make_save_names(save_path)

if os.path.isfile(save_path_cross):
    cross_mean = np.loadtxt(save_path_cross)
    cross_tr_mean = np.loadtxt(save_path_cross_tr)
    F_mean = np.loadtxt(save_path_f)
else:
    crosses_list = []
    crosses_tr_list = []
    F_measures_list = []
    sess,_ = model.load(save_path)
    for i in range(10):
        crosses,crosses_tr,F_measures = model.compute_eval_metrics_pred(data, target,seq_lens,0.5,None,sess=sess)
        crosses_list += [crosses]
        crosses_tr_list += [crosses_tr]
        F_measures_list += [F_measures]

    cross_mean = np.mean(crosses_list,axis=0)
    cross_tr_mean =np.mean(crosses_tr_list,axis=0)
    F_mean = np.mean(F_measures_list,axis=0)

    np.savetxt(save_path_cross,cross_mean)
    np.savetxt(save_path_cross_tr,cross_tr_mean)
    np.savetxt(save_path_f,F_mean)

crosses = [cross_mean]
crosses_tr = [cross_tr_mean]
F_measures = [F_mean]
model_names = [os.path.basename(save_path)]

if args.compare is not None:
    for save_path_compare in args.compare:
        save_path_cross,save_path_cross_tr,save_path_f = make_save_names(save_path_compare)
        if os.path.isfile(save_path_cross):
            cross_mean = np.loadtxt(save_path_cross)
            cross_tr_mean = np.loadtxt(save_path_cross_tr)
            F_mean = np.loadtxt(save_path_f)
        else:
            crosses_list = []
            crosses_tr_list = []
            F_measures_list = []
            sess,_ = model.load(save_path_compare)
            for i in range(10):
                crosses_comp,crosses_tr_comp,F_measures_comp = model.compute_eval_metrics_pred(data, target,seq_lens,0.5,None,sess=sess)
                crosses_list += [crosses_comp]
                crosses_tr_list += [crosses_tr_comp]
                F_measures_list += [F_measures_comp]
            cross_mean = np.mean(crosses_list,axis=0)
            cross_tr_mean =np.mean(crosses_tr_list,axis=0)
            F_mean = np.mean(F_measures_list,axis=0)

            np.savetxt(save_path_cross_tr,cross_mean)
            np.savetxt(save_path_cross_tr,cross_tr_mean)
            np.savetxt(save_path_f,F_mean)

        crosses += [cross_mean]
        crosses_tr += [cross_tr_mean]
        F_measures += [F_mean]
        model_names += [os.path.basename(save_path_compare)]


fig, [ax1,ax2,ax3] = plt.subplots(1,3)
# x = np.around(np.arange(0,1,0.1),1)
x_labels = np.around(np.arange(1,0,-0.1),1)

for i in range(len(crosses)):
    ax1.plot(x_labels[::-1],crosses[i],label=model_names[i])
    ax2.plot(x_labels[::-1],crosses_tr[i],label=model_names[i])
    ax3.plot(x_labels[::-1],F_measures[i],label=model_names[i])
for ax in [ax1,ax2,ax3]:
    ax.set_xticks(x_labels[::-1])
    ax.set_xticklabels(x_labels)
ax1.set_title('Cross-entropy')
ax2.set_title('Transition cross-entropy')
ax3.set_title('F-measure')
plt.legend()

plt.show()

# print(f"XE_GT: {result_GT[0]},XE_tr_GT: {result_GT[1]},F0_GT: {result_GT[2]}")
# print(f"XE_s: {result_s[0]},XE_tr_s: {result_s[1]},F0_s: {result_s[2]}")
# print(f"XE_th: {result_th[0]},XE_tr_th: {result_th[1]},F0_th: {result_th[2]}")
