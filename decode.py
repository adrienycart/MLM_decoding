import numpy as np
import itertools
import queue
import argparse
import pretty_midi
import sys
import pickle
import os
import warnings

import dataMaps
import eval_utils
from beam import Beam
from state import State
from mlm_training.model import Model, make_model_param


def decode(acoustic, model, sess, branch_factor=50, beam_size=200, union=False, weight=[[0.8], [0.2]],
           hash_length=10, out=None, history=5, weight_model=None, features=False, verbose=True,
           is_weight=True):
    """
    Transduce the given acoustic probabilistic piano roll into a binary piano roll.

    Parameters
    ==========
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

    weight : matrix
        A 2 x (1 or 88) matrix, whose first row is the weight for the acoustic model and whose 2nd
        row is the weight for the language model, either for each pitch (2x88) or across all pitches
        (2x1). Each column in the matrix should be normalized to sum to 1. Defaults to [[0.8], [0.2]].

    hash_length : int
        The history length for the hashed beam. If two states do not differ in the past hash_length
        frames, only the most probable one is saved in the beam. Defaults to 10.

    out : string
        The directory in which to save the outputs, or None to not save anything. Defaults to None.

    history : int
        This history length to use for the weight_model. Defaults to 5.

    weight_model : sklearn.model
        The sklearn model to use to set dynamic weights for the models. Defaults to None, which uses
        the static weight of the weight parameter.
        
    features : boolean
        Whether to use features in the weight_model's data points. Defaults to False.

    verbose : bool
        Print progress in number of frames.
        
    is_weight : bool
        True if the given weight_model outputs weights. False if it outputs weighted priors. Unused
        if weight_model is None. Defaults to False.


    Returns
    =======
    piano_roll : np.ndarray
        An 88 x T binary piano roll, where a 1 represents the presence of a pitch
        at a given frame.

    priors : np.ndarray
        An 88 x T matrix, giving the prior assigned to each pitch detection by the
        most probable language model state.
        
    weights : np.ndarray
        An 88 X T matrix, giving the acoustic weights for each pitch at each frame.
    """
    if union:
        branch_factor = int(branch_factor / 2)

    beam = Beam()
    beam.add_initial_state(model, sess)

    acoustic = np.transpose(acoustic)

    for frame_num, frame in enumerate(acoustic):
        if verbose and frame_num % 20 == 0:
            print(str(frame_num) + " / " + str(acoustic.shape[0]))

        states = []
        samples = []
        weights = []
        priors = []
        p = None

        if weight_model:
            X = np.vstack([create_weight_x(state, acoustic, frame_num, history, features=features) for state in beam])
            # 2 x len(X) matrix
            weights_all = np.transpose(weight_model.predict_proba(X)) if is_weight else np.zeros((2, len(X)))
            # len(X) array
            priors_all = np.squeeze(weight_model.predict_proba(X)[:, 1]) if not is_weight else np.zeros(len(X))
            if not is_weight:
                p = []

        # Used for union sampling
        unique_samples = []

        # Gather all computations to perform them batched
        # Acoustic sampling is done separately because the acoustic samples will be identical for every state.
        if union or (not weight_model and weight[0][0] == 1.0):
            # If sampling method is acoustic (or union), we generate the same samples for every current hypothesis
            for _, sample in itertools.islice(enumerate_samples(frame, beam.beam[0].prior,
                                              weight=[[1.0], [0.0]]), branch_factor):
                binary_sample = np.zeros(88)
                binary_sample[sample] = 1

                # This is used to check for overlaps in union case
                if union:
                    unique_samples.append(list(binary_sample))

                for i, state in enumerate(beam):
                    weight_this = weights_all[:, i * 88 : (i + 1) * 88] if weight_model else weight
                    states.append(state)
                    priors.append(np.squeeze(state.prior))
                    weights.append(weight_this)
                    samples.append(binary_sample)

        if union or weight_model or weight[0][0] != 1.0:
            for i, state in enumerate(beam):
                weight_this = weights_all[:, i * 88 : (i + 1) * 88] if weight_model and is_weight else weight
                sample_weight = [[0.0], [1.0]] if union else weight_this
                
                prior_this = priors_all[i * 88 : (i + 1) * 88] if (not is_weight) and weight_model else None
                
                for _, sample in itertools.islice(enumerate_samples(frame, state.prior, weight=sample_weight,
                                                  p=prior_this), branch_factor):

                    binary_sample = np.zeros(88)
                    binary_sample[sample] = 1

                    # Overlap with acoustic sample in union case. Skip this sample.
                    if not (union and list(binary_sample) in unique_samples):
                        priors.append(np.squeeze(state.prior))
                        states.append(state)
                        samples.append(binary_sample)
                        weights.append(weight_this)
                        if p is not None:
                            p.append(prior_this)

        log_probs, combined_priors = get_log_prob(np.array(samples), np.array(frame), np.array(priors),
                                                  np.array(weights), p=None if p is None else np.array(p))

        np_samples = np.zeros((len(samples), 1, 88))
        for i, sample in enumerate(samples):
            np_samples[i, 0, :] = sample

        hidden_states, priors = model.run_one_step([s.hidden_state for s in states], np_samples, sess)

        beam = Beam()
        for hidden_state, prior, log_prob, state, sample, w, combined_prior in zip(hidden_states, priors,
                                                                                   log_probs, states, samples,
                                                                                   weights, combined_priors):
            beam.add(state.transition(sample, log_prob, hidden_state, prior, w[0], combined_prior))

        beam.cut_to_size(beam_size, min(hash_length, frame_num + 1))

        if out and frame_num % 1 == 0:
            output = [(s.get_piano_roll(), s.get_priors(), s.get_weights(), s.get_combined_priors()) for s in beam]
            with open(os.path.join(out, 'data_' + str(frame_num) + '.pkl'), 'wb') as file:
                pickle.dump(output, file)

    top_state = beam.get_top_state()
    return top_state.get_piano_roll(), top_state.get_priors(), top_state.get_weights(), top_state.get_combined_priors()




def create_weight_x(state, acoustic, frame_num, history, pitch_range=0, pitches=range(88), features=False):
    """
    Get the x input for the dynamic weighting model.

    Parameters
    ==========
    state : State
        The state to examine for its piano roll and prior.

    acoustic : np.ndarray
        The acoustic prior for the entire piece.
        
    frame_num : int
        The current frame number.

    history : int
        How many frames to save in the x data point. Defaults to 5.
        
    pitch_range : int
        How many pitches above and below the given pitches should be saved in the data point.
        Defaults to 0.

    pitches : list
        The pitches we want data points for. Defaults to [0:88] (all pitches).
        
    features : boolean
        True to calculate features. False otherwise.

    Returns
    =======
    x : np.ndarray
        The x data points for the given input for the dynamic weighting model.
    """
    frame = acoustic[frame_num, :]
    if features:
        return np.hstack((state.get_piano_roll(min_length=history, max_length=history),
                          get_features(acoustic, frame_num, state.get_priors()), np.reshape(frame, (88, -1)),
                          np.reshape(state.prior, (88, -1))))[pitches]
        
    else:
        return np.hstack((state.get_piano_roll(min_length=history, max_length=history),
                          np.reshape(frame, (88, -1)), np.reshape(state.prior, (88, -1))))[pitches]


    
def get_features(acoustic, frame_num, priors):
    """
    Get a features array from the given acoustic and language model priors.
    
    Parameters
    ==========
    acoustic : np.ndarray
        The acoustic prior for the entire piece.
        
    frame_num : int
        The current frame number.
        
    language : np.ndarray
        The language priors from the entire piece.
        
    Returns
    =======
    features : np.ndarray
        A 88 x (num_features) array.
    """
    def uncertainty(array):
        """
        Get the average squared error of each number in an array's distance from certainty (1 or 0).
        
        Parameters
        ==========
        array : np.array
            Any data array.
            
        Returns
        =======
        uncertainty : float
            The average squared error of each element in the array's difference from certainty (1 or 0).
        """
        normed = np.minimum(array, np.abs(1 - array))
        return np.mean(normed * normed)

    def entropy(array):
        """
        Get the entropy of the given array.
        
        Parameters
        ==========
        array : np.array
            An data array.
            
        Returns
        =======
        entropy : float
            The entropy of the given array. A measure of its flatness.
        """
        return np.sum(np.where(array == 0, 0, -array * np.log2(array))) / np.log2(len(array))
    
    num_features = 8
    frame = acoustic[frame_num, :]
    language = np.squeeze(priors[:, -1])
    features = np.zeros((88, num_features))
    
    features[:, 0] = uncertainty(acoustic)
    features[:, 1] = uncertainty(language)
    features[:, 2] = entropy(acoustic)
    features[:, 3] = entropy(language)
    features[:, 4] = np.mean(acoustic)
    features[:, 5] = np.mean(language)
    
    if frame_num != 0:
        features[:, 6] = frame - acoustic[frame_num, :]
        features[:, 7] = language - priors[:, -2]
    else:
        features[:, 6] = frame
        features[:, 7] = language
    
    return features




def get_log_prob(sample, acoustic, language, weight, p=None):
    """
    Get the log probability of a set of samples given the priors and weights.

    Parameters
    ==========
    sample : np.ndarray
        An N x 88 matrix representing N possible samples.

    acoustic : np.array
        An 88-length array, containing the probability of each pitch being present,
        according to the acoustic model.

    language : np.ndarray
        An N x 88 matrix containing the probability of each pitch being present,
        according to each sampled state.

    weight : np.ndarray
        An N x 2 x (1 or 88) size tensor, whose first index corresponds to each of the N samples,
        second index corresponds to the prior (index 0 for acoustic prior, index 1 for language prior),
        and whose third dimension is either length 1 (when each pitch has the same prior) or length
        88 (when each pitch has a different prior).
        
    p : np.array
        A weighted probability prior for each pitch. This overrides all other arguments
        to be used as p if it is given. Defaults to None.

    Returns
    =======
    log_prob : np.array
        The log probability of each given sample, as a weighted sum of p(. | acoustic) and p(. | language),
        in an N x 1 array.
        
    combined_priors : np.ndarray
        The combined (NOT log) prior of each given sample, in an N x 88 nd-array.
    """
    if p is None:
        weight_acoustic = np.squeeze(weight[:, 0, :]) # N or N x 88
        weight_language = np.squeeze(weight[:, 1, :]) # N or N x 88

        if np.ndim(weight_acoustic) == 1:
            p = np.outer(weight_acoustic, acoustic) + language * np.reshape(weight_language, (-1, 1))
        else:
            p = weight_acoustic * acoustic + weight_language * language # N x 88
    else:
        p = np.squeeze(p)
        
    not_p = 1 - p

    return np.sum(np.log(np.where(sample == 1, p, not_p)), axis=1), p




def enumerate_samples(acoustic, language, weight=[[0.8], [0.2]], p=None):
    """
    Enumerate the binarised piano-roll samples of a frame, ordered by probability.

    Based on Algorithm 2 from:
    Boulanger-Lewandowski, Bengio, Vincent - 2013 - High-dimensional Sequence Transduction

    Parameters
    ==========
    acoustic : np.array
        An 88-length array, containing the probability of each pitch being present,
        according to the acoustic model.

    language : np.array
        An 88-length array, containing the probability of each pitch being present,
        according to the language model.

    weight : np.ndarray
        A 2 x (1 or 88) matrix, whose first row is the weight for the acoustic model and whose 2nd
        row is the weight for the language model, either for each pitch (2x88) or across all pitches
        (2x1). Each column in the matrix should be normalized to sum to 1. Defaults to [[0.8], [0.2]].
        
    p : np.array
        A weighted probability prior for each pitch. This overrides all other arguments
        to be used as p if it is given. Defaults to None.

    Return
    ======
    A generator for the given samples, ordered by probability, which will return
    the log-probability of a sample and the sample itself, a set of indices with an active pitch.
    """
    # set up p and not_p probabilities
    if p is None:
        p = np.squeeze(weight[0] * acoustic + weight[1] * language)
        not_p = np.squeeze(weight[0] * (1 - acoustic) + weight[1] * (1 - language))
    else:
        p = np.squeeze(p)
        not_p = 1 - p

    # Base case: most likely chosen greedily
    v_0 = np.where(p > not_p)[0]
    l_0 = np.sum(np.log(np.maximum(p, not_p)))
    yield l_0, v_0

    v_0 = set(v_0)

    # Sort likelihoods by likelihood penalty for flipping
    L = np.abs(np.log(p / not_p))
    R = L.argsort()
    L_sorted = L[R]

    # Num solves crash for duplicate likelihoods
    num = 1
    q = queue.PriorityQueue()
    q.put((L_sorted[0], 0, [0]))

    while not q.empty():
        l, _, v = q.get()
        yield l_0 - l, list(v_0.symmetric_difference(R[v]))

        i = np.max(v)
        if i + 1 < len(L_sorted):
            v.append(i + 1)
            q.put((l + L_sorted[i + 1], num, v))
            v = v.copy()
            
            # XOR between v and [i]
            try:
                v.remove(i)
            except:
                v.append(i)
                
            q.put((l + L_sorted[i + 1] - L_sorted[i], num+1, v))
            num += 2






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
    parser.add_argument("-wm", "--weight_model", help="Load the given sklearn model using pickle, to dynamically " +
                        "set weights. Defaults to None, which uses the static weight from -w instead.",
                        default=None)

    parser.add_argument("--max_len",type=str,help="test on the first max_len seconds of each text file. " +
                        "Anything other than a number will evaluate on whole files. Default is 30s.",
                        default=30)

    parser.add_argument("-o", "--output", help="The directory to save outputs to. Defaults to None (don't save).",
                        default=None)
    parser.add_argument("--hash", help="The hash length to use. Defaults to 10.",
                        type=int, default=10)
    parser.add_argument("-v", "--verbose", help="Print frame status updates.", action="store_true")

    args = parser.parse_args()

    if not (0 <= args.weight <= 1):
        print("Weight must be between 0 and 1.", file=sys.stderr)
        sys.exit(1)

    warnings.filterwarnings("ignore", message="tick should be an int.")

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

    # Load weight model
    weight_model = None
    history = None
    features = None
    is_weight = None
    if args.weight_model:
        with open(args.weight_model, "rb") as file:
            weight_model_dict = pickle.load(file)
            weight_model = weight_model_dict['model']
            history = weight_model_dict['history']
            if 'features' in weight_model_dict:
                features = weight_model_dict['features']
            else:
                features = False
            if 'weight' in weight_model_dict:
                is_weight = weight_model_dict['weight']
            else:
                is_weight = True

    # Decode
    pr, priors, weights, combined_priors = decode(data.input, model, sess, branch_factor=args.branch,
                         beam_size=args.beam, union=args.union, weight=[[args.weight], [1 - args.weight]],
                         out=args.output, hash_length=args.hash, history=history, weight_model=weight_model,
                         is_weight=is_weight, features=features, verbose=args.verbose)

    # Evaluate
    np.save("pr", pr)
    np.save("priors", priors)
    np.save("weights", weights)
    np.save("combined_priors", combined_priors)
    if args.step in ['quant','event']:
        pr = dataMaps.convert_note_to_time(pr, data.corresp, max_len=max_len)

    data = dataMaps.DataMaps()
    data.make_from_file(args.MIDI, "time", section=section)
    target = data.target

    P_f, R_f, F_f = eval_utils.compute_eval_metrics_frame(pr, target)
    P_n, R_n, F_n = eval_utils.compute_eval_metrics_note(pr, target, min_dur=0.05)

    print(f"Frame P,R,F: {P_f:.3f},{R_f:.3f},{F_f:.3f}, Note P,R,F: {P_n:.3f},{R_n:.3f},{F_n:.3f}")
