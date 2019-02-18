import numpy as np
import argparse
import pickle
import glob
import os
import hmmlearn

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from dataMaps import DataMaps
from eval_utils import compute_eval_metrics_frame, compute_eval_metrics_note


def decode(pr, priors, transitions):
    """
    Get the decoded piano roll given a trained HMM.
    
    Parameters
    ==========
    pr : np.ndarray
        An input piano roll with acoustic priors.
        
    priors : np.array
        An 88-length array with the prior for each pitch. That is, P(pitch | pitch_num).
        
    transitions : np.ndarray
        An 88x2x2 nd-array, which contains, for each pitch, its HMM's probabilities.
        transitions[p,i,j] refers to the probability, for pitch p, to transition from state i
        into state j. State 0 represents inactive while state 1 represents active.
    
    Returns
    =======
    decoded_pr : np.ndarray
        The output, binarized, decoded piano roll.
    """
    decoded_pr = np.zeros(pr.shape)
    
    # TODO with hmmlearn, pitch by pitch
    
    return decoded_pr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('data_path',type=str, help="folder containing the split dataset")
    parser.add_argument('hmm_probs', type=str, help="Location to load the HMM probabilites from.")
    
    parser.add_argument("--step", type=str, choices=["time", "quant", "event"],
                        help="Change the step type for frame timing. Either time (default), " +
                        "quant (for 16th notes), or event (for onsets).", default="time")
    
    parser.add_argument("--max_len", type=str, help="test on the first max_len seconds of each " + 
                        "text file. Anything other than a number will evaluate on whole files. Default is 30s.",
                        default=30)
    
    parser.add_argument("--save", help="location to save the computed results. If not provided, results are not saved.")

    args = parser.parse_args()
    
    try:
        max_len = float(args.max_len)
        section = [0, max_len]
    except:
        max_len = None
        section = None
    
    with open(args.output, "rb") as file:
        pkl = pickle.load(args.hmm_probs)
        
    priors = pkl["priors"]
    transitions = pkl["transitions"]
    
    results = {}
    frames = np.zeros((0, 3))
    notes = np.zeros((0, 3))
    
    for file in glob.glob(os.path.join(args.data_path, "*.mid")):
        print(file)
        sys.stdout.flush()
        
        data = DataMaps()
        data.make_from_file(file, args.step, section)
        
        pr = decode(data.input, priors, transitions)
        
        # Save output
        if not args.save is None:
            np.save(os.path.join(args.save, file.replace('.mid','_pr')), pr)
            np.savetxt(os.path.join(args.save, file.replace('.mid','_pr.csv')), pr)
        
        if args.step in ['quant','event']:
            pr = convert_note_to_time(pr, data.corresp, max_len=max_len)

        data = DataMaps()
        data.make_from_file(filename, "time", section=section)
        target = data.target

        #Evaluate
        P_f,R_f,F_f = compute_eval_metrics_frame(pr, target)
        P_n,R_n,F_n = compute_eval_metrics_note(pr, target, min_dur=0.05)

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
        pickle.dump(results, open(os.path.join(args.save, 'results.pkl'), "wb"))