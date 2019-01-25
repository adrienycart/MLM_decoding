from dataMaps import DataMaps,convert_note_to_time, align_matrix
from eval_utils import compute_eval_metrics_frame, compute_eval_metrics_note
from mlm_training.model import Model, make_model_param
from mlm_training.utils import safe_mkdir
from decode import decode

import os
import argparse
from datetime import datetime
import sys
import pickle

import tensorflow as tf
import pretty_midi as pm
import numpy as np




parser = argparse.ArgumentParser()

parser.add_argument('model',type=str,help="location of the checkpoint to load (inside ckpt folder)")
parser.add_argument('data_path',type=str,help="folder containing the split dataset")
parser.add_argument("--step", type=str, choices=["time", "quant", "event"], help="Change the step type for frame timing. Either time (default), " +
                    "quant (for 16th notes), or event (for onsets).", default="time")
parser.add_argument("--max_len",type=str,help="test on the first max_len seconds of each text file. Anything other than a number will evaluate on whole files. Default is 30s.",
                    default=30)
parser.add_argument('--save',type=str,help="location to save the computed results. If not provided, results are not saved")
parser.add_argument("-b", "--beam", type=int, help="The beam size. Defaults to 100.", default=100)
parser.add_argument("-k", "--branch", type=int, help="The branching factor. Defaults to 20.", default=20)
parser.add_argument("-u", "--union", help="Use the union sampling method.", action="store_true")
parser.add_argument("-w", "--weight", help="The weight for the acoustic model (between 0 and 1). " +
                    "Defaults to 0.5", type=float, default=0.5)
parser.add_argument("--hash", help="The hash length to use. Defaults to 10.", type=int, default=10)


args = parser.parse_args()

if not (0 <= args.weight <= 1):
    print("Weight must be between 0 and 1.", file=sys.stderr)
    sys.exit(2)

try:
    max_len = float(args.max_len)
    section = [0,max_len]
    print(f"Evaluate on first {args.max_len} seconds")
except:
    max_len = None
    section=None
    print(f"Evaluate on whole files")

note_range = [21,109]
note_min = note_range[0]
note_max = note_range[1]

n_hidden = 256



# Load model
model_param = make_model_param()
model_param['n_hidden'] = n_hidden
model_param['n_steps'] = 1 # To generate 1 step at a time

# Build model object
model = Model(model_param)
sess,_ = model.load(args.model, model_path=args.model)

if not args.save is None:
    safe_mkdir(args.save)

results = {}
folder = args.data_path
frames = np.zeros((0, 3))
notes = np.zeros((0, 3))

for fn in os.listdir(folder):
    if fn.endswith('.mid') and not fn.startswith('.'):
        filename = os.path.join(folder,fn)
        print(filename)

        data = DataMaps()
        data.make_from_file(filename,args.step,section)

        # Decode
        pr, priors = decode(data.input, model, sess, branch_factor=args.branch, beam_size=args.beam,
                        union=args.union, weight=[args.weight, 1 - args.weight], hash_length=args.hash)
        #pr = (data.input>0.5).astype(int)

        # Save output
        if not args.save is None:
            np.savetxt(os.path.join(args.save,fn.replace('.mid','_pr.csv')), pr)
            np.savetxt(os.path.join(args.save,fn.replace('.mid','_priors.csv')), priors)

        if args.step in ['quant','event']:
            pr = convert_note_to_time(pr,data.corresp,max_len=max_len)

        data = DataMaps()
        data.make_from_file(filename, "time", section=section)
        target = data.target
            
        #Evaluate
        P_f,R_f,F_f = compute_eval_metrics_frame(pr,target)
        P_n,R_n,F_n = compute_eval_metrics_note(pr,target,min_dur=0.05)
        
        frames = np.vstack((frames, [P_f, R_f, F_f]))
        notes = np.vstack((frames, [P_n, R_n, F_n]))

        print(f"Frame P,R,F: {P_f:.3f},{R_f:.3f},{F_f:.3f}, Note P,R,F: {P_n:.3f},{R_n:.3f},{F_n:.3f}")

        results[fn] = [[P_f,R_f,F_f],[P_n,R_n,F_n]]

P_f, R_f, F_f = np.mean(frames, axis=0)
P_n, R_n, F_n = np.mean(notes, axis=0)
print(f"Averages: Frame P,R,F: {P_f:.3f},{R_f:.3f},{F_f:.3f}, Note P,R,F: {P_n:.3f},{R_n:.3f},{F_n:.3f}")
        
if not args.save is None:
    pickle.dump(results,open(os.path.join(args.save,'results.p'), "wb"))