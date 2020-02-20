import numpy as np
import argparse
import pickle
import glob
import os

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from dataMaps import DataMaps, DataMapsBeats, convert_note_to_time
from eval_utils import compute_eval_metrics_frame, compute_eval_metrics_note




def decode(observed, prior, trans, order=None):
    """
    Get the most likely state sequence given the parameters.

    Parameters
    ==========
    observed : np.array
        The observed sequence of floats.

    prior : np.array
        The prior for each state.

    trans : np.ndarray
        A (2 ** order)x2 matrix representing the transition probabilities as trans[from, to].

    order : int
        The order of the hmm to decode. None to calculate it from trans's dimensionality.

    Returns
    =======
    decoded : np.array
        A binarized sequence of the same length as the observed sequence representing
        the most likely state at each observed data point.

    prob : float
        The log probability of the returned sequence.
    """
    # Convert all to log probabilities to avoid underflow
    observed = np.log(np.stack((1 - observed, observed), axis=1))
    prior = np.log(prior)
    trans = np.log(trans)

    if order is None:
        order = 0
        power = len(trans)
        while power != 1:
            power /= 2
            order += 1

    order_mask = 2 ** order - 1

    if order != 1:
        new_prior = np.zeros(len(trans))
        new_prior[:2] = prior
        prior = new_prior

    # Set up save lists
    state_probs = np.zeros((len(observed), len(trans)))
    state_paths = np.zeros((len(observed), len(trans)))

    initial_observed = np.zeros(len(trans))
    initial_observed[:2] = observed[0, :]

    state_probs[0, :] = prior + initial_observed
    state_paths[0, :] = np.zeros(len(trans), dtype=int)

    # Decode
    for frame in range(1, len(observed)):

        max_probs = np.ones(len(trans)) * -np.inf
        max_paths = np.zeros(len(trans), dtype=int)

        for from_state in range(len(trans)):
            for to_index in range(2):
                prob = state_probs[frame-1, from_state] + trans[from_state, to_index] + observed[frame, to_index]

                to_state = ((from_state << 1) | to_index) & order_mask

                if prob > max_probs[to_state]:
                    max_probs[to_state] = prob
                    max_paths[to_state] = from_state

        state_probs[frame, :] = max_probs
        state_paths[frame, :] = max_paths

    # Get most likely decoded sequence
    decoded = np.zeros(len(observed), dtype=int)
    decoded[-1] = np.argmax(state_probs[-1, :])

    # Walk backwards
    for frame in range(len(observed) - 1, 0, -1):
        decoded[frame-1] = state_paths[frame, decoded[frame]]

    return decoded & 1, np.max(state_probs[-1, :])




def decode_all_pitches(pr, priors, transitions):
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

    for pitch in range(88):
        observed = np.squeeze(pr[pitch])
        prior = [1 - priors[pitch], priors[pitch]]
        trans = np.squeeze(transitions[pitch, :, :])

        decoded_pr[pitch, :], _ = decode(observed, prior, trans)

    return decoded_pr



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('data_path',type=str, help="folder containing the split dataset")
    parser.add_argument('hmm', type=str, help="Location to load the HMM probabilites from.")

    parser.add_argument("--step", type=str, choices=["time", "quant", "event",'beat'],
                        help="Change the step type for frame timing. Either time (default), " +
                        "quant (for 16th notes), or event (for onsets).", default="time")
    parser.add_argument('--beat_gt',action='store_true',help="with beat timesteps, use ground-truth beat positions")
    parser.add_argument('--beat_subdiv',type=str,help="with beat timesteps, beat subdivisions to use (comma separated list, without brackets)",default='0,1/4,1/3,1/2,2/3,3/4')

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

    with open(args.hmm, "rb") as file:
        pkl = pickle.load(file)

    if args.save is not None:
        os.makedirs(model_dir, exist_ok=True)

    priors = pkl["priors"]
    transitions = pkl["transitions"]

    results = {}
    frames = np.zeros((0, 3))
    notes = np.zeros((0, 3))

    for file in glob.glob(os.path.join(args.data_path, "*.mid")):
        base = os.path.basename(file)
        print(base)
        sys.stdout.flush()

        if args.step == "beat":
            data = DataMapsBeats()
            data.make_from_file(file,args.beat_gt,args.beat_subdiv,section, acoustic_model="kelz")
        else:
            data = DataMaps()
            data.make_from_file(file, args.step, section,acoustic_model="kelz")


        pr = decode_all_pitches(data.input, priors, transitions)

        # Save output
        if not args.save is None:
            np.save(os.path.join(args.save, base.replace('.mid','_pr')), pr)
            np.savetxt(os.path.join(args.save, base.replace('.mid','_pr.csv')), pr)

        if args.step in ['quant','event','beat']:
            pr = convert_note_to_time(pr, data.corresp, data.input_fs, max_len=max_len)

        data = DataMaps()
        data.make_from_file(file, "time", section=section, acoustic_model="kelz")
        target = data.target

        #Evaluate
        P_f,R_f,F_f = compute_eval_metrics_frame(pr, target)
        P_n,R_n,F_n = compute_eval_metrics_note(pr, target, min_dur=0.05)

        frames = np.vstack((frames, [P_f, R_f, F_f]))
        notes = np.vstack((notes, [P_n, R_n, F_n]))

        print(f"Frame P,R,F: {P_f:.3f},{R_f:.3f},{F_f:.3f}, Note P,R,F: {P_n:.3f},{R_n:.3f},{F_n:.3f}")
        sys.stdout.flush()

        results[base] = [[P_f,R_f,F_f],[P_n,R_n,F_n]]

    P_f, R_f, F_f = np.mean(frames, axis=0)
    P_n, R_n, F_n = np.mean(notes, axis=0)
    print(f"Averages: Frame P,R,F: {P_f:.3f},{R_f:.3f},{F_f:.3f}, Note P,R,F: {P_n:.3f},{R_n:.3f},{F_n:.3f}")
    sys.stdout.flush()

    if not args.save is None:
        pickle.dump(results, open(os.path.join(args.save, 'results.pkl'), "wb"))
