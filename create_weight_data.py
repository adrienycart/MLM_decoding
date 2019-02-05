import argparse
import dataMaps
import decode
import numpy as np
import os
import pickle
import itertools
import glob

from beam import Beam
from mlm_training.model import Model, make_model_param




def get_weight_data(gt, acoustic, model, sess, branch_factor=50, beam_size=200, union=False, weight=[[0.5], [0.5]],
           hash_length=10, gt_only=False, history=5, min_diff=0.01, features=False, verbose=False):
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
        
    gt_only : boolean
        True to transition only on the ground truth sample, no matter its rank. Flase to transition
        normally. Defaults to False.
        
    history : int
        How many frames to save in the x data point. Defaults to 5.
        
    min_diff : float
        The minimum difference (between language and acoustic) to save a data point. Defaults to 0.01.
        
    features : boolean
        Whether to use features in the weight_model's data points. Defaults to False.
        
    
    Returns
    =======
    x : np.ndarray
        The x data from this decoding process. A (data x 7) size matrix.
        
    y : np.array
        The y data from this decoding process. A data-length array.
    """
    if union:
        branch_factor = int(branch_factor / 2)
    
    x = None
    y = np.zeros(0)
    
    beam = Beam()
    beam.add_initial_state(model, sess)
    
    gt = np.transpose(gt)
    ranks = []
    
    acoustic = np.transpose(acoustic)
    
    for frame_num, frame in enumerate(acoustic):
        if frame_num % 20 == 0 and verbose:
            print(str(frame_num) + " / " + str(acoustic.shape[0]))
        gt_frame = gt[frame_num, :]
        
        states = []
        samples = []
        weights = []
        priors = []
        
        # Used for union sampling
        unique_samples = []
        
        # Get data
        for state in beam:
            pitches = np.argwhere(1 - np.isclose(np.squeeze(state.prior), np.squeeze(frame),
                                                 rtol=0.0, atol=min_diff))[:,0] if min_diff > 0 else np.arange(88)
            if len(pitches) > 0:
                if x is not None:
                    x = np.vstack((x, decode.create_weight_x(state, acoustic, frame_num, history, pitches=pitches,
                                                             features=features)))
                else:
                    x = decode.create_weight_x(state, acoustic, frame_num, history, pitches=pitches, features=features)
                y = np.append(y, gt_frame[pitches])
        
        # Gather all computations to perform them batched
        # Acoustic sampling is done separately because the acoustic samples will be identical for every state.
        if gt_only:
            states = [beam.get_top_state()]
            samples = [gt_frame]
            weights = [[[1.0], [0.0]]]
            priors = [np.squeeze(states[0].prior)]
            
        else:
            if union or weight[0][0] == 1.0:
                # If sampling method is acoustic (or union), we generate the same samples for every current hypothesis
                for _, sample in itertools.islice(decode.enumerate_samples(frame, beam.beam[0].prior,
                                                  weight=[[1.0], [0.0]]), branch_factor):
                    binary_sample = np.zeros(88)
                    binary_sample[sample] = 1

                    # This is used to check for overlaps in union case
                    if union:
                        unique_samples.append(list(binary_sample))

                    for i, state in enumerate(beam):
                        weight_this = weight
                        states.append(state)
                        priors.append(np.squeeze(state.prior))
                        weights.append(weight_this)
                        samples.append(binary_sample)

            if union or weight[0][0] != 1.0:
                for i, state in enumerate(beam):
                    sample_weight = [[0.0], [1.0]] if union else weight
                    for _, sample in itertools.islice(decode.enumerate_samples(frame, state.prior,
                                                      weight=sample_weight), branch_factor):

                        binary_sample = np.zeros(88)
                        binary_sample[sample] = 1

                        # Overlap with acoustic sample in union case. Skip this sample.
                        if not (union and list(binary_sample) in unique_samples):
                            weight_this = weight

                            priors.append(np.squeeze(state.prior))
                            states.append(state)
                            samples.append(binary_sample)
                            weights.append(weight_this)

        log_probs, combined_priors = decode.get_log_prob(np.array(samples), np.array(frame),
                                                         np.array(priors), np.array(weights))

        np_samples = np.zeros((len(samples), 1, 88))
        for i, sample in enumerate(samples):
            np_samples[i, 0, :] = sample

        hidden_states, priors = model.run_one_step([s.hidden_state for s in states], np_samples, sess)

        beam = Beam()
        for hidden_state, prior, log_prob, state, sample, w, combined_prior in zip(hidden_states, priors,
                                                                                   log_probs, states, samples,
                                                                                   weights, combined_priors):
            beam.add(state.transition(sample, log_prob, hidden_state, prior, w, combined_prior))

        beam.cut_to_size(beam_size, min(hash_length, frame_num + 1))
        
    return x, y




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("MIDI", help="The MIDI file to load, or a directory containing MIDI files. " +
                        "They should be in the same location as the " +
                        "corresponding acoustic csvs, and have the same name (except the extension).")
    
    parser.add_argument("--out", help="The file to save data.", required=True)
    
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
    
    parser.add_argument("--max_len",type=str,help="test on the first max_len seconds of each text file. " +
                        "Anything other than a number will evaluate on whole files. Default is 30s.",
                        default=30)
    
    parser.add_argument("--hash", help="The hash length to use. Defaults to 10.",
                        type=int, default=10)
    
    parser.add_argument("--history", help="The history length to use. Defaults to 5.",
                        type=int, default=5)
    
    parser.add_argument("--min_diff", help="The minimum difference (between language and acoustic) to " +
                        "save a data point.", type=float, default=0.01)
    
    parser.add_argument("--gt", help="Transition on ground truth samples only.", action="store_true")
    
    parser.add_argument("--features", help="Use features in the x data points.", action="store_true")
    
    parser.add_argument("-v", "--verbose", help="Print frame updates", action="store_true")
    
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
    
    # Load model
    model_param = make_model_param()
    model_param['n_hidden'] = args.hidden
    model_param['n_steps'] = 1 # To generate 1 step at a time

    # Build model object
    model = Model(model_param)
    sess,_ = model.load(args.model, model_path=args.model)
    
    # Load data
    if args.MIDI.endswith(".mid"):
        data = dataMaps.DataMaps()
        data.make_from_file(args.MIDI, args.step, section=section)
    
        # Decode
        X, Y = get_weight_data(data.target, data.input, model, sess, branch_factor=args.branch, beam_size=args.beam,
                               union=args.union, weight=[args.weight, 1 - args.weight], hash_length=args.hash,
                               gt_only=args.gt, history=args.history, features=args.features, min_diff=args.min_diff,
                               verbose=args.verbose)
    else:
        X = None
        Y = np.zeros(0)
        
        for file in glob.glob(os.path.join(args.MIDI, "*.mid")):
            if args.verbose:
                print(file)
            data = dataMaps.DataMaps()
            data.make_from_file(file, args.step, section=section)

            # Decode
            x, y = get_weight_data(data.target, data.input, model, sess, branch_factor=args.branch, beam_size=args.beam,
                                   union=args.union, weight=[[args.weight], [1 - args.weight]], hash_length=args.hash,
                                   gt_only=args.gt, history=args.history, features=args.features, min_diff=args.min_diff,
                                   verbose=args.verbose)
            
            if X is not None:
                X = np.vstack((X, x))
            else:
                X = x
            Y = np.append(Y, y)
    
    print(X.shape)
    print(Y.shape)
    
    # Save data
    with open(args.out, "wb") as file:
              pickle.dump({'X' : X,
                           'Y' : Y,
                           'history' : args.history,
                           'features' : args.features}, file)
    