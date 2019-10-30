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
timestep.add_argument('-quant_short',action='store_true',help="use quant_short timesteps")
parser.add_argument('-compare',type=str,nargs='*',help="compare with following models (can have several values)")
parser.add_argument('-sched_mix',action='store_true',help='evaluate by sampling from acoustic outputs')
parser.add_argument('-diagRNN',action='store_true',help='use diagLSTM units')
parser.add_argument('-no_sched',action='store_true',help='compute the results without scheduled sampling')
parser.add_argument('-no_save',action='store_true',help='do not load, do not save results')
parser.add_argument('-no_chunks',action='store_true',help='do not cut sequences into chunks')
parser.add_argument('-plot',action='store_true',help='plot outputs of all compared models')
args = parser.parse_args()


if args.quant:
    timestep_type = 'quant'
    max_len = 300
elif args.event:
    timestep_type = 'event'
    max_len = 100
elif args.quant_short:
    timestep_type = 'quant_short'
    max_len = 900
else:
    timestep_type = 'time'
    max_len = 750

if args.no_chunks:
    max_len = 60

note_range = [21,109]
note_min = note_range[0]
note_max = note_range[1]


n_hidden = 256 #number of features in hidden layer

rolls_dict = {}


def make_save_names(save_path):
    if args.sched_mix:
        cross_path = os.path.join('ckpt',save_path,'result_cross_mix.txt')
        cross_tr_path = os.path.join('ckpt',save_path,'result_cross_tr_mix.txt')
        F_path = os.path.join('ckpt',save_path,'result_F_mix.txt')
        S_path = os.path.join('ckpt',save_path,'result_S_mix.txt')
    else:
        cross_path = os.path.join('ckpt',save_path,'result_cross.txt')
        cross_tr_path = os.path.join('ckpt',save_path,'result_cross_tr.txt')
        F_path = os.path.join('ckpt',save_path,'result_F.txt')
        S_path = os.path.join('ckpt',save_path,'result_S.txt')
    return [cross_path,cross_tr_path,F_path,S_path]

save_path = args.save_path


all_save_names = sum([make_save_names(path) for path in [args.save_path]+args.compare],[])
if not all([os.path.isfile(path) for path in all_save_names]) or args.no_save:
    #If not all data has been compute already
    if args.sched_mix:
        data= DatasetMaps()
        data.load_data(args.data_path,timestep_type=timestep_type,subsets=['test'],acoustic_model='bittner')

    else:
        data = Dataset()
        if args.no_chunks:
            data.load_data_one(args.data_path,subset='test',timestep_type=timestep_type,note_range=note_range,max_len=max_len)
            data.zero_pad()
        else:
            data.load_data_one(args.data_path,subset='test',timestep_type=timestep_type,note_range=note_range)
    data.note_range = note_range

    model_param = make_model_param()
    model_param['n_hidden']=n_hidden
    model_param['learning_rate']=0
    if not args.no_chunks:
        model_param['chunks']=max_len
    if args.sched_mix:
        model_param['scheduled_sampling'] = "mix"
        model_param['sampl_mix_weight'] = 1.0
    else:
        model_param['scheduled_sampling'] = "self"
    if args.diagRNN:
        model_param['cell_type'] = 'diagLSTM'

    model = make_model_from_dataset(data,model_param)
    model.print_params()
    model.build_graph()

    if args.no_chunks:
        dataset, target, seq_lens, keys = data.get_dataset('test',with_keys=True)
    else:
        dataset, target, seq_lens, keys = data.get_dataset_chunks_no_pad('test',max_len,with_keys=True)





save_path_cross,save_path_cross_tr,save_path_f,save_path_s = make_save_names(save_path)



if os.path.isfile(save_path_cross) and not args.no_save:
    cross_mean = np.loadtxt(save_path_cross)
    cross_tr_mean = np.loadtxt(save_path_cross_tr)
    F_mean = np.loadtxt(save_path_f)
    S_mean = np.loadtxt(save_path_s)
else:
    crosses_list = []
    crosses_tr_list = []
    F_measures_list = []
    S_list = []
    sess,_ = model.load(save_path)

    if args.no_sched:
        repeats = [1]
    else:
        repeats = range(10)
    for i in repeats:
        crosses,crosses_tr,F_measures,Scores = model.compute_eval_metrics_pred(dataset, target,seq_lens,0.5,None,keys=keys,sess=sess,no_sched=args.no_sched)
        crosses_list += [crosses]
        crosses_tr_list += [crosses_tr]
        F_measures_list += [F_measures]
        S_list += [Scores]

        if args.plot:
            for pr in data.test:
                roll = np.array([pr.roll[:,:-1]])
                pred = model.run_prediction(roll,[pr.length],None,sigmoid=True,sess=sess)
                rolls_dict[pr.name] = {}
                rolls_dict[pr.name]['input']=roll[0]
                rolls_dict[pr.name]['pred_'+save_path]=pred[0]





    cross_mean = np.mean(crosses_list,axis=0)
    cross_tr_mean =np.mean(crosses_tr_list,axis=0)
    F_mean = np.mean(F_measures_list,axis=0)
    S_mean = np.mean(S_list,axis=0)

    if not  args.no_save:
        np.savetxt(save_path_cross,cross_mean)
        np.savetxt(save_path_cross_tr,cross_tr_mean)
        np.savetxt(save_path_f,F_mean)
        np.savetxt(save_path_s,S_mean)

crosses_comp = [cross_mean]
crosses_tr_comp = [cross_tr_mean]
F_measures_comp = [F_mean]
Scores_comp = [S_mean]
model_names = [os.path.basename(save_path)]

if args.compare is not None:
    for save_path_compare in args.compare:
        save_path_cross,save_path_cross_tr,save_path_f,save_path_s = make_save_names(save_path_compare)
        if os.path.isfile(save_path_cross) and not args.no_save:
            cross_mean = np.loadtxt(save_path_cross)
            cross_tr_mean = np.loadtxt(save_path_cross_tr)
            F_mean = np.loadtxt(save_path_f)
            S_mean = np.loadtxt(save_path_s)
        else:
            crosses_list = []
            crosses_tr_list = []
            F_measures_list = []
            S_list = []
            sess,_ = model.load(save_path_compare)
            if args.no_sched:
                repeats = [1]
            else:
                repeats = range(10)
            for i in repeats:
                crosses,crosses_tr,F_measures,Scores = model.compute_eval_metrics_pred(dataset, target,seq_lens,0.5,None,keys=keys,sess=sess,no_sched=args.no_sched)
                crosses_list += [crosses]
                crosses_tr_list += [crosses_tr]
                F_measures_list += [F_measures]
                S_list += [Scores]

            if args.plot:
                for pr in data.test:
                    roll = np.array([pr.roll[:,:-1]])
                    pred = model.run_prediction(roll,[pr.length],None,sigmoid=True,sess=sess)
                    rolls_dict[pr.name]['pred_'+save_path_compare]=pred[0]


            cross_mean = np.mean(crosses_list,axis=0)
            cross_tr_mean =np.mean(crosses_tr_list,axis=0)
            F_mean = np.mean(F_measures_list,axis=0)
            S_mean = np.mean(S_list,axis=0)

            if not args.no_save:
                np.savetxt(save_path_cross,cross_mean)
                np.savetxt(save_path_cross_tr,cross_tr_mean)
                np.savetxt(save_path_f,F_mean)
                np.savetxt(save_path_s,S_mean)

        crosses_comp += [cross_mean]
        crosses_tr_comp += [cross_tr_mean]
        F_measures_comp += [F_mean]
        Scores_comp += [S_mean]

        model_names += [os.path.basename(save_path_compare)]




if args.no_sched:
    for name, XE, XE_tr, F, S in zip(model_names,crosses_comp,crosses_tr_comp,F_measures_comp,Scores_comp):
        print(name)
        print('XE:', XE)
        print('XE_tr:', XE_tr)
        print('F:', F)
        print('S:', S)

else:
    fig, [ax1,ax2,ax3,ax4] = plt.subplots(1,4)
    # x = np.around(np.arange(0,1,0.1),1)
    x_labels = np.around(np.arange(1,0,-0.1),1)

    for i in range(len(crosses_comp)):
        ax1.plot(x_labels[::-1],crosses_comp[i],label=model_names[i])
        ax2.plot(x_labels[::-1],crosses_tr_comp[i],label=model_names[i])
        ax3.plot(x_labels[::-1],F_measures_comp[i],label=model_names[i])
        ax4.plot(x_labels[::-1],Scores_comp[i],label=model_names[i])
    for ax in [ax1,ax2,ax3,ax4]:
        ax.set_xticks(x_labels[::-1])
        ax.set_xticklabels(x_labels)
    ax1.set_title('Cross-entropy')
    ax2.set_title('Transition cross-entropy')
    ax3.set_title('F-measure')
    ax4.set_title('Score')
    plt.legend()

    plt.show()


if args.plot:
    import matplotlib.pyplot as plt
    for name,prs in rolls_dict.items():

        # print(roll.shape,pred1.shape,pred2.shape)

        fig, axes = plt.subplots(len(prs),1,figsize=(12,6))
        axes[0].imshow(prs['input'][:,:400],origin='lower',aspect='auto')
        axes[0].set_title('Input: '+name)
        for path,ax in zip([args.save_path]+args.compare,axes[1:]):
            ax.imshow(prs['pred_'+path][:,:400],origin='lower',aspect='auto')
            ax.set_title(path)
        plt.show()



# print(f"XE_GT: {result_GT[0]},XE_tr_GT: {result_GT[1]},F0_GT: {result_GT[2]}")
# print(f"XE_s: {result_s[0]},XE_tr_s: {result_s[1]},F0_s: {result_s[2]}")
# print(f"XE_th: {result_th[0]},XE_tr_th: {result_th[1]},F0_th: {result_th[2]}")
