import skopt
import argparse
import os
import sys
import pickle
import numpy as np
import glob

import hmm_eval

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from dataMaps import DataMaps, DataMapsBeats, convert_note_to_time
from eval_utils import compute_eval_metrics_frame, compute_eval_metrics_note

order = None
priors = None
step = None
with_offset = None
beat_gt = None
beat_subdiv = None
data_path = None

def test(params):
    global priors
    global step
    global beat_gt
    global beat_subdiv
    global order
    global with_offset
    global data_path
    print(params)

    transitions = np.zeros((88, 2 ** order, 2))
    for i in range(88):
        transitions[i, :, 0] = params
        transitions[i, :, 1] = 1 - transitions[i, :, 0]

    frames = np.zeros((0, 3))
    notes = np.zeros((0, 3))

    folder = data_path
    for file in glob.glob(os.path.join(folder, "*.mid")):
        print(file)
        sys.stdout.flush()

        if step == "beat":
            data = DataMapsBeats()
            data.make_from_file(file,beat_gt,beat_subdiv,[0,30], acoustic_model='kelz')
        else:
            data = DataMaps()
            data.make_from_file(file, step, [0,30],acoustic_model='kelz')

        pr = hmm_eval.decode_all_pitches(data.input, priors, transitions)

        if step != "time":
            pr = convert_note_to_time(pr, data.corresp, data.input_fs, max_len=30)

        data = DataMaps()
        data.make_from_file(file, "time", section=[0, 30], acoustic_model="kelz")
        target = data.target

        #Evaluate
        P_f,R_f,F_f = compute_eval_metrics_frame(pr, target)
        P_n,R_n,F_n = compute_eval_metrics_note(pr, target, min_dur=0.05, with_offset=with_offset)

        print(f"Frame P,R,F: {P_f:.3f},{R_f:.3f},{F_f:.3f}, Note P,R,F: {P_n:.3f},{R_n:.3f},{F_n:.3f}")
        sys.stdout.flush()

        frames = np.vstack((frames, [P_f, R_f, F_f]))
        notes = np.vstack((notes, [P_n, R_n, F_n]))

    P_f, R_f, F_f = np.mean(frames, axis=0)
    P_n, R_n, F_n = np.mean(notes, axis=0)

    print(f"Frame P,R,F: {P_f:.3f},{R_f:.3f},{F_f:.3f}, Note P,R,F: {P_n:.3f},{R_n:.3f},{F_n:.3f}")
    print(str(F_n) + ": " + str(params))
    sys.stdout.flush()

    out = "hmm/models/" + step + "." + str(order) + "." + str(F_n) + ".pkl"
    with open(out, "wb") as file:
        pickle.dump({"priors" : priors,
                     "transitions" : transitions}, file)


    return -F_n



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("priors", help="The pickle file to get the hmm priors from.")
    parser.add_argument('data_path',type=str, help="folder containing the split dataset")
    parser.add_argument("--step", type=str, choices=["time", "quant", "event",'beat'],
                        help="Change the step type for frame timing. Either time (default), " +
                        "quant (for 16th notes), or event (for onsets).", default="time")
    parser.add_argument('--beat_gt',action='store_true',help="with beat timesteps, use ground-truth beat positions")
    parser.add_argument('--beat_subdiv',type=str,help="with beat timesteps, beat subdivisions to use (comma separated list, without brackets)",default='0,1/4,1/3,1/2,2/3,3/4')
    parser.add_argument("-o", "--output", help="The file to save the resulting optimization to. Defaults to optim.sko.",
                        default="optim.sko")
    parser.add_argument("--iters", help="The number of iterations to run optimization for (after the initial 10 " +
                        "random points). Defaults to 200.", type=int, default=200)
    parser.add_argument("--order", help="The order of the HMM to train (unigram, bigram, etc)", type=int, default=1)
    parser.add_argument("--with_offset", help="Train the HMM for F-measure with offsets.", action="store_true")
    args = parser.parse_args()

    with open(args.priors, "rb") as file:
        priors = pickle.load(file)["priors"]

    step = args.step
    beat_gt = args.beat_gt
    beat_subdiv = args.beat_subdiv
    order = args.order
    with_offset = args.with_offset
    data_path = args.data_path

    print("Running for " + str(args.iters) + " iterations.")
    print("step type: " + args.step)
    print("order: " + str(args.order))
    print("With offsets" if args.with_offset else "Without offsets")
    print("saving output to " + args.output)
    sys.stdout.flush()

    if args.output is not None:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

    dimensions = [(0.0, 1.0) for i in range(2 ** order)] # probability of transition into state 0 for all states of all pitches

    opt = skopt.gp_minimize(test, dimensions, n_calls=10+args.iters, verbose=True)

    skopt.dump(opt, args.output)
