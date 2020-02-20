import numpy as np
import argparse
import pickle
import glob
import os

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from dataMaps import DataMaps


def get_pitch_priors(prs):
    """
    Get the pitch priors (with add-1 smoothing) given a list of piano rolls.

    Parameters
    ==========
    prs : list(np.ndarray)
        A list of piano rolls, from which we will generate the priors for each pitch.

    Returns
    =======
    priors : np.array
        An 88-length array with the prior for each pitch. That is, P(pitch | pitch_num).
    """
    priors = np.ones(88)
    num_frames = 1

    for pr in prs:
        num_frames += pr.shape[1]

        priors += np.sum(pr, axis=1)

    return priors / num_frames


def get_transition_probs(prs):
    """
    Get the transition probabilities (with add-1 smoothing) for each pitch HMM's states.

    Parameters
    ==========
    prs : list(np.ndarray)
        A list of piano rolls, from which we will generate the priors for each pitch.

    Returns
    =======
    transitions : np.ndarray
        An 88x2x2 nd-array, which contains, for each pitch, its HMM's probabilities.
        transitions[p,i,j] refers to the probability, for pitch p, to transition from state i
        into state j. State 0 represents inactive while state 1 represents active.
    """
    transitions = np.ones((88, 2, 2))

    for pr in prs:
        for frame in range(pr.shape[1] - 1):
            for pitch in range(88):
                transitions[pitch, int(pr[pitch, frame]), int(pr[pitch, frame+1])] += 1

    return transitions / np.sum(transitions, axis=2, keepdims=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('data_path',type=str, help="folder containing the split dataset")
    parser.add_argument('output', type=str, help="Location to save the computed results.")
    parser.add_argument("--step", type=str, choices=["time", "quant", "event",'beat'],
                        help="Change the step type for frame timing. Either time (default), " +
                        "quant (for 16th notes), or event (for onsets).", default="time")
    parser.add_argument('--beat_gt',action='store_true',help="with beat timesteps, use ground-truth beat positions")
    parser.add_argument('--beat_subdiv',type=str,help="with beat timesteps, beat subdivisions to use (comma separated list, without brackets)",default='0,1/4,1/3,1/2,2/3,3/4')
    parser.add_argument("--max_len", type=str, help="test on the first max_len seconds of each " +
                        "text file. Anything other than a number will evaluate on whole files. Default is 30s.",
                        default=30)

    parser.add_argument('--acoustic_model', type=str, help='acoustic model to use',default='kelz')

    args = parser.parse_args()

    try:
        max_len = float(args.max_len)
        section = [0, max_len]
    except:
        max_len = None
        section = None

    print("Reading MIDI files")

    prs = []

    for file in glob.glob(os.path.join(args.data_path, "*.mid")):

        if args.step == "beat":
            data = DataMapsBeats()
            data.make_from_file(filename,args.beat_gt,args.beat_subdiv,section, acoustic_model=args.acoustic_model)
        else:
            data = DataMaps()
            data.make_from_file(file, args.step, section,acoustic_model=args.acoustic_model)

        prs.append(data.target)

    print("Calculating priors and transition probabilities")

    with open(args.output, "wb") as file:
        pickle.dump({"priors" : get_pitch_priors(prs),
                     "transitions" : get_transition_probs(prs)}, file)
