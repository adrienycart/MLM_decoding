import numpy as np
import itertools
import queue
import argparse
import pretty_midi
import sys
import pickle
import os
import decode

import dataMaps
import eval_utils
from beam import Beam
from state import State
from mlm_training.model import Model, make_model_param


def get_gt_rank(gt, acoustic, model, sess, branch_factor=50, beam_size=200, union=False, weight=[0.5, 0.5],
           hash_length=10, gt_max=None, gt_only=False):
    """
    Get the average ranks of the ground truth frame from decode.enumerate_samples().
    
    Parameters
    ==========
    gt : matrix
        The ground truth binary piano roll, 88 x T.
    
    acoustic : matrix
        A probabilistic piano roll, 88 x T, containing values between 0.0 and 1.0
        inclusive. acoustic[p, t] represents the probability of pitch p being present
        at frame t.
        
    model : Model
        The language model to use for the transduction process.
        
    sess : tf.session
        The session for the given model.
        
    branch_factor : int
        The number of samples to use per frame. Defaults to 50.
        
    beam_size : int
        The beam size for the search. Defaults to 50.
        
    union : boolean
        True to use union sampling. False (default) to use joint sampling with the weight.
        
    weight : list
        A length-2 list, whose first element is the weight for the acoustic model and whose 2nd
        element is the weight for the language model. This list should be normalized to sum to 1.
        Defaults to [0.5, 0.5].
        
    hash_length : int
        The history length for the hashed beam. If two states do not differ in the past hash_length
        frames, only the most probable one is saved in the beam. Defaults to 10.
        
    gt_max : int
        The maximum rank to check for the ground truth sample. Defaults to None (no limit).
        
    gt_only : boolean
        True to transition only on the ground truth sample, no matter its rank. Flase to transition
        normally. Defaults to False.
        
    
    Returns
    =======
    ranks : list
        A list of the ranks of the ground truth sample for each transition.
    """
    if union:
        branch_factor = int(branch_factor / 2)
    
    beam = Beam()
    beam.add_initial_state(model, sess)
    
    gt = np.transpose(gt)
    ranks = []
    
    if gt_max is not None and union:
        gt_max = int(gt_max / 2)
    
    for frame_num, frame in enumerate(np.transpose(acoustic)):
        print(str(frame_num) + " / " + str(acoustic.shape[1]))
        gt_frame = np.nonzero(gt[frame_num, :])[0]
        
        states = []
        samples = []
        log_probs = []
        
        # Used for union sampling
        unique_samples = []
        
        # Gather all computations to perform them batched
        # Acoustic sampling is done separately because the acoustic samples will be identical for every state.
        if union or weight[0] == 1.0:
            # If sampling method is acoustic (or union), we generate the same samples for every current hypothesis
            rank_ac, enumerated_samples = get_rank_and_samples(gt_frame, frame, beam.beam[0].prior,
                                                            [1.0, 0.0], 0 if gt_only else branch_factor, gt_max)
            if not union:
                ranks.append(rank_ac)
                
            if gt_only:
                enumerated_samples = [gt_frame]
                
            for sample in enumerated_samples:
                binary_sample = np.zeros(88)
                binary_sample[sample] = 1
                
                # This is used to check for overlaps in union case
                if union:
                    unique_samples.append(list(binary_sample))
                
                for state in beam:
                    states.append(state)
                    samples.append(binary_sample)
                    log_probs.append(decode.get_log_prob(binary_sample, frame, state.prior, weight))
            
        if union or weight[0] != 1.0:
            for state in beam:
                rank_la, enumerated_samples = get_rank_and_samples(gt_frame, frame, state.prior,
                                                            [0.0, 1.0] if union else weight,
                                                            0 if gt_only else branch_factor, gt_max)
                
                if union:
                    ranks.append(min(rank_ac, rank_la))
                else:
                    ranks.append(rank_la)
                    
                if gt_only and not union:
                    enumerated_samples = [gt_frame]
                    
                for sample in enumerated_samples:
                    binary_sample = np.zeros(88)
                    binary_sample[sample] = 1

                    # Overlap with acoustic sample in union case. Skip this sample.
                    if not (union and list(binary_sample) in unique_samples):
                        states.append(state)
                        samples.append(binary_sample)
                        log_probs.append(decode.get_log_prob(binary_sample, frame, state.prior, weight))
                
        np_samples = np.zeros((len(samples), 1, 88))
        for i, sample in enumerate(samples):
            np_samples[i, 0, :] = sample
        
        hidden_states, priors = model.run_one_step([s.hidden_state for s in states], np_samples, sess)
        
        beam = Beam()
        for hidden_state, prior, log_prob, state, sample in zip(hidden_states, priors, log_probs, states, samples):
            beam.add(state.transition(sample, log_prob, hidden_state, prior))
        
        beam.cut_to_size(beam_size, min(hash_length, frame_num + 1))
        
    return ranks




def get_rank_and_samples(gt, acoustic, language, weight, branch_factor, gt_max):
    """
    Get the rank of the ground truth, and enumerate the samples of a frame by probability.
    
    Parameters
    ==========
    gt : np.ndarray
        An 88-length array, the ground truth sample.
        
    acoustic : np.ndarray
        An 88-length array, containing the probability of each pitch being present,
        according to the acoustic model.
        
    language : np.ndarray
        An 88-length array, containing the probability of each pitch being present,
        according to the language model.
        
    weight : list
        A length-2 list, whose first element is the weight for the acoustic model and whose 2nd
        element is the weight for the language model. This list should be normalized to sum to 1.
        
    branch_factor : int
        The number of samples to return in samples.
        
    gt_max : int
        The maximum rank to check for the ground truth sample. Defaults to None (no limit).
        
    Return
    ======
    rank : int
        The rank of the ground truth sample.
        
    samples : list
        The samples, ordered by probability.
    """
    rank = None
    samples = []
    
    num = 0
    for _, sample in decode.enumerate_samples(acoustic, language, weight):
        if (gt_max is None or num <= gt_max) and rank is None and np.array_equal(sample, gt):
            rank = num
            
        if len(samples) < branch_factor:
            samples.append(sample)
            
        if len(samples) == branch_factor and (rank is not None or (gt_max is not None and gt_max <= num)):
            return rank, samples
        
        num = num + 1
        
    return None, None


            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("MIDI", help="The MIDI file to load. This should be in the same location as the " +
                        "corresponding acoustic csv, and have the same name (except the extension).")
    
    parser.add_argument("-m", "--model", help="The location of the trained language model.", required=True)
    parser.add_argument("--hidden", help="The number of hidden layers in the language model. Defaults to 256",
                        type=int, default=256)
    
    parser.add_argument("--step", type=str, choices=["time", "quant", "event"], help="Change the step type " +
                        "for frame timing. Either time (default), quant (for 16th notes), or event (for onsets).",
                        default="time")
    
    parser.add_argument("-b", "--beam", help="The beam size. Defaults to 100.", type=int, default=100)
    parser.add_argument("-k", "--branch", help="The branching factor. Defaults to 20.", type=int, default=20)
    
    parser.add_argument("-u", "--union", help="Use the union sampling method.", action="store_true")
    parser.add_argument("-w", "--weight", help="The weight for the acoustic model (between 0 and 1). " +
                        "Defaults to 0.5", type=float, default=0.5)
    
    parser.add_argument("--max_len", type=str, help="test on the first max_len seconds of each text file. " +
                        "Anything other than a number will evaluate on whole files. Default is 30s.",
                        default=30)
    
    parser.add_argument("--hash", help="The hash length to use. Defaults to 10.",
                        type=int, default=10)
    
    parser.add_argument("--gt", help="Transition on ground truth samples only.", action="store_true")
    parser.add_argument("--gt_max", type=int, help="The maximum rank to check for the ground truth sample.",
                        default=None)
    
    args = parser.parse_args()
        
    if not (0 <= args.weight <= 1):
        print("Weight must be between 0 and 1.", file=sys.stderr)
        sys.exit(1)
        
    try:
        max_len = float(args.max_len)
        section = [0, max_len]
    except:
        max_len = None
        section = None
    
    # Load data
    data = dataMaps.DataMaps()
    data.make_from_file(args.MIDI, args.step, section=section)
    
    # Load model
    model_param = make_model_param()
    model_param['n_hidden'] = args.hidden
    model_param['n_steps'] = 1 # To generate 1 step at a time

    # Build model object
    model = Model(model_param)
    sess,_ = model.load(args.model, model_path=args.model)
    
    # Decode
    ranks = get_gt_rank(data.target, data.input, model, sess, branch_factor=args.branch, beam_size=args.beam,
                        union=args.union, weight=[args.weight, 1 - args.weight], hash_length=args.hash,
                        gt_max=args.gt_max, gt_only=args.gt)
    
    print(ranks)
