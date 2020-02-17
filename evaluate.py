from dataMaps import DataMaps, DataMapsBeats, convert_note_to_time, align_matrix
from eval_utils import compute_eval_metrics_frame, compute_eval_metrics_note, compute_eval_metrics_with_onset, make_midi_from_notes
from mlm_training.model import Model, make_model_param
from mlm_training.utils import safe_mkdir
from decode import decode

import os
import argparse
from datetime import datetime
import sys
import pickle
import warnings

import tensorflow as tf
from tensorflow import keras
import pretty_midi as pm
import numpy as np




parser = argparse.ArgumentParser()

parser.add_argument('model',type=str,help="location of the checkpoint to load (inside ckpt folder)")
parser.add_argument('data_path',type=str,help="folder containing the split dataset")
parser.add_argument("--step", type=str, choices=["time","20ms", "quant","quant_short", "event","beat"], help="Change the step type for frame timing. Either time (default), " +
                    "quant (for 16th notes), or event (for onsets).", default="time")
parser.add_argument('--beat_gt',action='store_true',help="with beat timesteps, use ground-truth beat positions")
parser.add_argument('--beat_subdiv',type=str,help="with beat timesteps, beat subdivisions to use (comma separated list, without brackets)",default='0,1/4,1/3,1/2,2/3,3/4')
parser.add_argument("--max_len",type=str,help="test on the first max_len seconds of each text file. Anything other than a number will evaluate on whole files. Default is 30s.",
                    default=30)
parser.add_argument('--save',type=str,help="location to save the computed results. If not provided, results are not saved")
parser.add_argument("-b", "--beam", type=int, help="The beam size. Defaults to 50.", default=50)
parser.add_argument("-k", "--branch", type=int, help="The branching factor. Defaults to 5.", default=5)
weight = parser.add_mutually_exclusive_group()
weight.add_argument("-w", "--weight", help="The weight for the acoustic model (between 0 and 1). " +
                    "If -1, priors are multiplied (only with --it). Defaults to 0.5", type=float, default=0.5)
weight.add_argument("-wm", "--weight_model", help="Load the given sklearn model using pickle, to dynamically " +
                    "set weights. Defaults to None, which uses the static weight from -w instead.",
                    default=None)
parser.add_argument("--hash", help="The hash length to use. Defaults to 12.", type=int, default=12)
parser.add_argument("-v", "--verbose", help="Use verbose printing.", action="store_true")
parser.add_argument("--gpu", help="The gpu to use. Defaults to 0.", default="0")
parser.add_argument("--gt", help="Use the gt to use the best possible weight_model results.", action="store_true")
parser.add_argument('--n_hidden', help="Number of hidden nodes for the LSTM", type=int, default=256)
parser.add_argument('--with_offset', help="use offset for framewise metrics", action='store_true')
parser.add_argument('--with_onsets', help="use presence/onset piano-roll", action='store_true')
parser.add_argument("--diagRNN", help="Use diagonal RNN units", action="store_true")
parser.add_argument("--merge_onsets", help="When there are onsets in consecutive frames, only take into account the first one", action="store_true")

args = parser.parse_args()

if not (0 <= args.weight <= 1):
    if args.weight == -1 :
        print('No weight - priors multiplied')
    else:
        print("Weight must be between 0 and 1.", file=sys.stderr)
        sys.exit(2)

print('####################################')

try:
    max_len = float(args.max_len)
    section = [0,max_len]
    print(f"Evaluate on first {args.max_len} seconds")
except:
    max_len = None
    section=None
    print(f"Evaluate on whole files")

print(f"Step: {args.step}")
if args.step == "beat":
    print(f"Use GT beats: {args.beat_gt}")
    print(f"Beat subdivisions: {args.beat_subdiv}")
print(f"Beam size: {args.beam}")
print(f"Branching factor: {args.branch}")
print(f"Hash size: {args.hash}")
if args.weight_model is None:
    print(f"Weight: {args.weight}")
else:
    print(f"Auto-weight: {args.weight_model}")
print(f"Sampling union: False")

print('####################################')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

warnings.filterwarnings("ignore", message="tick should be an int.")

note_range = [21,109]
note_min = note_range[0]
note_max = note_range[1]

model = None
sess = None
if args.weight_model is not None or args.weight != 1.0:
    n_hidden = args.n_hidden

    # Load model
    model_param = make_model_param()
    model_param['n_hidden'] = n_hidden
    model_param['n_steps'] = 1 # To generate 1 step at a time
    model_param['pitchwise']=False
    if args.diagRNN:
        model_param['cell_type'] = "diagLSTM"
    model_param['with_onsets'] = args.with_onsets

    # Build model object
    model = Model(model_param)
    sess,_ = model.load(args.model, model_path=args.model)

# Load weight model
weight_model_dict=None
weight_model=None
if args.weight_model is not None:
    with open(args.weight_model, "rb") as file:
        weight_model_dict = pickle.load(file)
    if 'model' in weight_model_dict:
        weight_model = weight_model_dict['model']
    else:
        weight_model = keras.models.load_model(weight_model_dict['model_path'])

if not args.save is None:
    safe_mkdir(args.save)

results = {}
folder = args.data_path
frames = np.zeros((0, 3))
notes = np.zeros((0, 3))

for fn in os.listdir(folder):
    if fn.endswith('.mid') and not fn.startswith('.'):# and not 'chpn-e01' in fn:
        filename = os.path.join(folder,fn)
        print(filename)
        sys.stdout.flush()

        if args.step == "beat":
            data = DataMapsBeats()
            data.make_from_file(filename,args.beat_gt,args.beat_subdiv,section, with_onsets=args.with_onsets, acoustic_model='kelz')
        else:
            data = DataMaps()
            data.make_from_file(filename,args.step,section, with_onsets=args.with_onsets, acoustic_model='kelz')

        input_data = data.input
        if args.with_onsets:
            input_data = np.zeros((data.input.shape[0] * 2, data.input.shape[1]))
            input_data[:data.input.shape[0], :] = data.input[:, :, 0]
            input_data[data.input.shape[0]:, :] = data.input[:, :, 1]

        pr, priors, weights, combined_priors = decode(input_data, model, sess, branch_factor=args.branch,
                        beam_size=args.beam, weight=[[args.weight], [1 - args.weight]],
                        out=None, hash_length=args.hash, weight_model_dict=weight_model_dict,
                        verbose=args.verbose, gt=data.target if args.gt else None, weight_model=weight_model)

        # Save output
        if not args.save is None:
            np.save(os.path.join(args.save,fn.replace('.mid','_pr')), pr)
            np.savetxt(os.path.join(args.save,fn.replace('.mid','_pr.csv')), pr)
            if (args.weight_model is not None or args.weight != 1.0):
                np.save(os.path.join(args.save,fn.replace('.mid','_priors')), priors)
                np.save(os.path.join(args.save,fn.replace('.mid','_weights')), weights)
                np.save(os.path.join(args.save,fn.replace('.mid','_combined_priors')), combined_priors)
                np.savetxt(os.path.join(args.save,fn.replace('.mid','_priors.csv')), priors)


        if args.with_onsets:

            # import matplotlib.pyplot as plt
            # plt.subplot(311)
            # plt.imshow(pr,aspect='auto',origin='lower')
            #
            # plt.subplot(312)
            # pr_2 = pr[:88,:]
            # pr_2[pr[88:,:]==1] = 2
            # plt.imshow(pr_2,aspect='auto',origin='lower')
            #
            #
            # plt.subplot(313)
            # plt.imshow(data.target,aspect='auto',origin='lower')
            # plt.show()


            target_data = pm.PrettyMIDI(filename)
            corresp = data.corresp
            [P_f,R_f,F_f],[P_n,R_n,F_n],notes_est,intervals_est = compute_eval_metrics_with_onset(pr,corresp,target_data,double_roll=True,min_dur=0.05,with_offset=args.with_offset,section=section,merge_consecutive_onsets=args.merge_onsets)

            if args.save:
                midi_data = make_midi_from_notes(notes_est, intervals_est)
                midi_data.write(os.path.join(args.save,fn))



        else:

            if args.step in ['quant','event','quant_short','beat']:
                pr = convert_note_to_time(pr,data.corresp,data.input_fs,max_len=max_len)

            data = DataMaps()
            if args.step == "20ms" or args.with_onsets:
                data.make_from_file(filename, "20ms", section=section, with_onsets=args.with_onsets, acoustic_model="kelz")
            else:
                data.make_from_file(filename, "time", section=section, with_onsets=args.with_onsets, acoustic_model="kelz")
            target = data.target
            #Evaluate
            P_f,R_f,F_f = compute_eval_metrics_frame(pr,target)
            P_n,R_n,F_n = compute_eval_metrics_note(pr,target,min_dur=0.05,with_offset=args.with_offset)

        frames = np.vstack((frames, [P_f, R_f, F_f]))
        notes = np.vstack((notes, [P_n, R_n, F_n]))

        print(f"Frame P,R,F: {P_f:.3f},{R_f:.3f},{F_f:.3f}, Note P,R,F: {P_n:.3f},{R_n:.3f},{F_n:.3f}")
        sys.stdout.flush()

        results[fn] = [[P_f,R_f,F_f],[P_n,R_n,F_n]]

P_f, R_f, F_f = np.mean(frames, axis=0)
P_n, R_n, F_n = np.mean(notes, axis=0)
print(f"Averages: Frame P,R,F: {P_f:.3f},{R_f:.3f},{F_f:.3f}, Note P,R,F: {P_n:.3f},{R_n:.3f},{F_n:.3f}")
sys.stdout.flush()

if not args.save is None:
    pickle.dump(results,open(os.path.join(args.save,'results.p'), "wb"))
