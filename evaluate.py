from dataMaps import DataMaps, DataMapsBeats, convert_note_to_time, align_matrix
from eval_utils import compute_eval_metrics_frame, compute_eval_metrics_note
from mlm_training.model import Model, make_model_param
from mlm_training.utils import safe_mkdir
from decode import decode, decode_pitchwise_iterative, decode_with_onsets

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
parser.add_argument("--step", type=str, choices=["time", "quant","quant_short", "event","beat"], help="Change the step type for frame timing. Either time (default), " +
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
parser.add_argument("--pitchwise", type=int, help="use pitchwise language model. Value is the number of semitones above and below current pitch to take into account.")
parser.add_argument("--it", help="Use iterative pitchwise processing with this number of iterations. " +
                    "Defaults to 0, which doesn't use iterative processing.", type=int, default=0)
parser.add_argument("--uncertainty", help="Add some uncertainty to the LSTM decoding outputs, when " +
                    "used with --it. The outputs will be scaled to a range of size " +
                    "(1 - 2*uncertainty), centered around 0.5. Specifically, " +
                    "(0.0, 1.0) -> (0.0+uncertainty, 1.0-uncertainty). Defaults to 0.0.",
                    type=float, default=0)
parser.add_argument('--n_hidden', help="Number of hidden nodes for the LSTM", type=int, default=256)
parser.add_argument('--with_offset', help="use offset for framewise metrics", action='store_true')
parser.add_argument('--with_onsets', help="use presence/onset piano-roll", action='store_true')
parser.add_argument("--diagRNN", help="Use diagonal RNN units", action="store_true")

args = parser.parse_args()

if not (0 <= args.weight <= 1):
    if args.weight == -1 and not args.it is None:
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
print(f"Pitchwise window: {args.pitchwise}")

print('####################################')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

warnings.filterwarnings("ignore", message="tick should be an int.")

note_range = [21,109]
note_min = note_range[0]
note_max = note_range[1]

if args.weight_model is not None or args.weight != 1.0:
    n_hidden = args.n_hidden

    # Load model
    model_param = make_model_param()
    model_param['n_hidden'] = n_hidden
    model_param['n_steps'] = 1 # To generate 1 step at a time
    if args.pitchwise is None:
        model_param['pitchwise']=False
    else:
        model_param['pitchwise']=True
        model_param['n_notes'] = 2*args.pitchwise+1
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

        # Decode
        if args.it > 0:
            prs = decode_pitchwise_iterative(data.input, model, sess, beam_size=args.beam,
                                             weight=[[args.weight], [1 - args.weight]],
                                             hash_length=args.hash, verbose=args.verbose, num_iters=args.it,
                                             uncertainty=args.uncertainty)

            pr = prs[-1]

        elif args.weight_model is not None or args.weight != 1.0:
            if args.with_onsets:
                pr, priors, weights, combined_priors = decode_with_onsets(data.input, model, sess, branch_factor=args.branch,
                            beam_size=args.beam, weight=[[args.weight], [1 - args.weight]],
                            out=None, hash_length=args.hash, weight_model_dict=weight_model_dict,
                            verbose=args.verbose, gt=data.target if args.gt else None, weight_model=weight_model)
            else:
                pr, priors, weights, combined_priors = decode(data.input, model, sess, branch_factor=args.branch,
                            beam_size=args.beam, weight=[[args.weight], [1 - args.weight]],
                            out=None, hash_length=args.hash, weight_model_dict=weight_model_dict,
                            verbose=args.verbose, gt=data.target if args.gt else None, weight_model=weight_model,
                            with_onsets=args.with_onsets)
        else:
            pr = (data.input>0.5).astype(int)

        # Save output
        if not args.save is None:
            np.save(os.path.join(args.save,fn.replace('.mid','_pr')), pr)
            np.savetxt(os.path.join(args.save,fn.replace('.mid','_pr.csv')), pr)
            if (args.weight_model is not None or args.weight != 1.0) and args.it <= 0:
                np.save(os.path.join(args.save,fn.replace('.mid','_priors')), priors)
                np.save(os.path.join(args.save,fn.replace('.mid','_weights')), weights)
                np.save(os.path.join(args.save,fn.replace('.mid','_combined_priors')), combined_priors)
                np.savetxt(os.path.join(args.save,fn.replace('.mid','_priors.csv')), priors)

        if args.step in ['quant','event','quant_short','beat']:
            pr = convert_note_to_time(pr,data.corresp,data.input_fs,max_len=max_len)

        data = DataMaps()
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
