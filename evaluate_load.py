import numpy as np
import os
from eval_utils import compute_eval_metrics_frame, compute_eval_metrics_note, out_key_errors_binary_mask
from mlm_training.utils import safe_mkdir
from dataMaps import DataMaps, DataMapsBeats, convert_note_to_time, align_matrix
import pickle
import argparse
import sys


parser = argparse.ArgumentParser()
parser.add_argument('input_folder',type=str)
parser.add_argument('target_folder',type=str)
parser.add_argument("--step", type=str, choices=["time", "quant","quant_short", "event","beat"], help="Change the step type for frame timing. Either time (default), " +
                    "quant (for 16th notes), or event (for onsets).", default="time")
parser.add_argument('--beat_gt',action='store_true',help="with beat timesteps, use ground-truth beat positions")
parser.add_argument('--beat_subdiv',type=str,help="with beat timesteps, beat subdivisions to use (comma separated list, without brackets)",default='0,1/4,1/3,1/2,2/3,3/4')
parser.add_argument('--with_offset', help="use offset for framewise metrics", action='store_true')
parser.add_argument('--with_quant',help="post-quantise the outputs",action='store_true')
parser.add_argument('--gap', help="Fill gaps <50ms.", action="store_true")
parser.add_argument('--save', help="save values to destination", type=str)

args = parser.parse_args()

input_folder = args.input_folder
target_folder = args.target_folder
# save_folder = 'results/baseline/raw'

frame = []
note = []
out_key = []

results = {}

if args.save is not None:
    safe_mkdir(args.save)

for fn in os.listdir(input_folder):
    if fn.endswith('_pr.csv') and not fn.startswith('.') and not 'chpn-e01' in fn:
        filename_input = os.path.join(input_folder,fn)
        filename_target = os.path.join(target_folder,fn).replace('_pr.csv','.mid')
        print(filename_input)

        data = DataMaps()
        data.make_from_file(filename_target,'time',[0,30],acoustic_model='kelz')

        input_roll = np.loadtxt(filename_input)
        target_roll = data.target
        mask = data.get_key_profile()
        mask_octave = data.get_key_profile_octave()

        # import matplotlib.pyplot as plt
        # plt.imshow(mask, aspect='auto')
        # plt.show(block=[bool])


        if args.step in ['quant','quant_short','event']:
            data_quant = DataMaps()
            data_quant.make_from_file(filename_target,args.step,[0,30],acoustic_model='kelz')
            input_roll = convert_note_to_time(input_roll,data_quant.corresp,data_quant.input_fs,max_len=30)
        if args.step == 'beat':
            data_quant = DataMapsBeats()
            data_quant.make_from_file(filename,args.beat_gt,args.beat_subdiv,section, acoustic_model='kelz')
            input_roll = convert_note_to_time(input_roll,data_quant.corresp,data_quant.input_fs,max_len=30)
        if args.step == 'time' and args.with_quant:
            data_quant = DataMaps()
            data_quant.make_from_file(filename_target,'quant',[0,30],acoustic_model='kelz')
            input_roll = align_matrix(input_roll,data_quant.corresp,data_quant.input_fs,method='quant')
            input_roll = convert_note_to_time(input_roll,data_quant.corresp,data_quant.input_fs,max_len=30)

        P_f,R_f,F_f = compute_eval_metrics_frame(input_roll,target_roll)
        P_n,R_n,F_n = compute_eval_metrics_note(input_roll,target_roll,min_dur=0.05,with_offset=args.with_offset,min_gap=0.05 if args.gap else None)
        err_FP, err_tot, err_FP_o, err_tot_o = out_key_errors_binary_mask(input_roll,target_roll,mask, mask_octave)



        print(f"Frame P,R,F: {P_f:.3f},{R_f:.3f},{F_f:.3f}, Note P,R,F: {P_n:.3f},{R_n:.3f},{F_n:.3f}")
        print(f"Out-key-errors FP: {err_FP:.3f}, Total: {err_tot:.3f}, OctaveFP: {err_FP_o:.3f}, OctaveTotal: {err_tot_o:.3f}")
        frame  += [[P_f,R_f,F_f]]
        note   += [[P_n,R_n,F_n]]
        out_key += [[err_FP,err_tot,err_FP_o, err_tot_o ]]


        results[fn.replace('_pr.csv','.mid')] = [[P_f,R_f,F_f],[P_n,R_n,F_n]]
        # print([[P_f,R_f,F_f],[P_n,R_n,F_n]])

print(np.array(frame).shape)
P_f, R_f, F_f = np.mean(frame, axis=0)
P_n, R_n, F_n = np.mean(note, axis=0)
err_FP, err_tot, err_FP_o, err_tot_o= np.mean(out_key, axis=0)

print(f"Averages: Frame P,R,F: {P_f:.3f},{R_f:.3f},{F_f:.3f}, Note P,R,F: {P_n:.3f},{R_n:.3f},{F_n:.3f}")
print(f"Averages: Out-key-errors FP: {err_FP:.3f}, Total: {err_tot:.3f}, OctaveFP: {err_FP_o:.3f}, OctaveTotal: {err_tot_o:.3f}")
sys.stdout.flush()

if args.save is not None:
    pickle.dump(results,open(args.save, "wb"))
