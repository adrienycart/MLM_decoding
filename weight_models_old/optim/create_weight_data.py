import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../..')

import argparse
import dataMaps
import decode
import numpy as np
import pickle
import itertools
import glob
import gzip

from beam import Beam
from mlm_training.model import Model, make_model_param




def get_weight_data(gt, acoustic, model, sess, branch_factor=50, beam_size=200, union=False, weight=[[0.5], [0.5]],
           hash_length=10, gt_only=False, history=5, min_diff=0.01, features=False, verbose=False, no_mlm=False):
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

    no_mlm : boolean
        Whether to suppress mlm-based inputs. Defaults to False.


    Returns
    =======
    x : np.ndarray
        The x data from this decoding process. A (data x 7) size matrix.

    y : np.array
        The y data from this decoding process. A data-length array.

    diffs : np.array
        The differences between the language and acoustic model priors for each data point.
    """
    weights_all = None
    priors_all = None

    beam = Beam()
    beam.add_initial_state(model, sess)

    acoustic = np.transpose(acoustic)

    x = np.zeros((0, 0))
    y = np.zeros(0)
    diffs = np.zeros(0)

    gt = np.transpose(gt)

    for frame_num, (gt_frame, frame) in enumerate(zip(gt, acoustic)):
        if verbose and frame_num % 20 == 0:
            print(str(frame_num) + " / " + str(acoustic.shape[0]))

        # Run the LSTM!
        if frame_num != 0:
            decode.run_lstm(sess, model, beam)

        # Here, beam contains a list of states, with sample histories, priors, and LSTM hidden_states,
        # but needs to be updated with weights and combined_priors when sampling.

        # Get data
        for state in beam:
            pitches = np.argwhere(1 - np.isclose(np.squeeze(state.prior), np.squeeze(frame),
                                                 rtol=0.0, atol=min_diff))[:,0] if min_diff > 0 else np.arange(88)

            if len(pitches) > 0:
                if len(x) > 0:
                    x = np.vstack((x, decode.create_weight_x_sk(state, acoustic, frame_num, history, pitches=pitches,
                                                                features=features, no_mlm=no_mlm, with_onsets=model.with_onsets)))
                else:
                    x = decode.create_weight_x_sk(state, acoustic, frame_num, history, pitches=pitches, features=features,
                                                  no_mlm=no_mlm, with_onsets=model.with_onsets)
                y = np.append(y, gt_frame[pitches])
                diffs = np.append(diffs, np.abs(np.squeeze(frame)[pitches] - np.squeeze(state.prior)[pitches]))

        new_beam = Beam()

        # Here we sample from each state in the beam
        if gt_only:
            new_beam.add(state.transition(gt_frame, 0.0))

        else:
            for i, state in enumerate(beam):
                weight_this = weights_all[:, i * 88 : (i + 1) * 88] if weights_all is not None else weight

                if priors_all is not None:
                    prior = np.squeeze(priors_all[i * 88 : (i + 1) * 88])
                else:
                    prior = np.squeeze(weight_this[0] * frame + weight_this[1] * state.prior)

                # Update state
                state.update_from_weight_model(weight_this[0], prior)

                for log_prob, sample in itertools.islice(decode.enumerate_samples(prior), branch_factor):

                    # Binarize the sample (return from enumerate_samples is an array of indexes)
                    binary_sample = np.zeros(88)
                    binary_sample[sample] = 1

                    # Transition on sample
                    new_beam.add(state.transition(binary_sample, log_prob))

        new_beam.cut_to_size(beam_size, min(hash_length, frame_num + 1))
        beam = new_beam

    return x, y, diffs




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("MIDI", help="The MIDI file to load, or a directory containing MIDI files. " +
                        "They should be in the same location as the " +
                        "corresponding acoustic csvs, and have the same name (except the extension).")

    parser.add_argument("--out", help="The file to save data.", required=True)

    parser.add_argument("-m", "--model", help="The location of the trained language model.", required=True)
    parser.add_argument("--hidden", help="The number of hidden layers in the language model. Defaults to 256",
                        type=int, default=256)

    parser.add_argument("--step", type=str, choices=["time", "quant","quant_short", "event", "beat"], help="Change the step type " +
                        "for frame timing. Either time (default), quant (for 16th notes), or event (for onsets).",
                        default="time")
    parser.add_argument('--beat_gt',action='store_true',help="with beat timesteps, use ground-truth beat positions")
    parser.add_argument('--beat_subdiv',type=str,help="with beat timesteps, beat subdivisions to use (comma separated list, without brackets)",default='0,1/4,1/3,1/2,2/3,3/4')

    parser.add_argument("--acoustic", type=str, choices=["kelz", "bittner"], help="Change the acoustic model " +
                        "used in the files. Either kelz (default), or bittner.",
                        default="kelz")

    parser.add_argument("-b", "--beam", help="The beam size. Defaults to 100.", type=int, default=100)
    parser.add_argument("-k", "--branch", help="The branching factor. Defaults to 20.", type=int, default=20)

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

    parser.add_argument("--no_mlm", help="Set all MLM-based inputs to 0.", action="store_true")

    parser.add_argument("--gpu", help="The GPU to use. Defaults to 0.", default="0")

    parser.add_argument("-v", "--verbose", help="Print frame updates", action="store_true")

    parser.add_argument("--diagRNN", help="Use diagonal RNN units", action="store_true")
    
    parser.add_argument("--with_onsets", help="The input will be a double pianoroll containing "
                        "presence and onset halves.", action="store_true")

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

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Load model
    model_param = make_model_param()
    model_param['n_hidden'] = args.hidden
    model_param['n_steps'] = 1 # To generate 1 step at a time
    if args.diagRNN:
        model_param['cell_type'] = "diagLSTM"
    model_param['with_onsets'] = args.with_onsets

    # Build model object
    model = Model(model_param)
    sess,_ = model.load(args.model, model_path=args.model)

    # Load data
    if args.MIDI.endswith(".mid"):
        if args.step == "beat":
            data = dataMaps.DataMapsBeats()
            data.make_from_file(args.MIDI,args.beat_gt,args.beat_subdiv,section, with_onsets=args.with_onsets, acoustic_model=args.acoustic)
        else:
            data = dataMaps.DataMaps()
            data.make_from_file(args.MIDI, args.step, section=section, with_onsets=args.with_onsets, acoustic_model=args.acoustic)

        input_data = data.input
        if args.with_onsets:
            input_data = np.zeros((data.input.shape[0] * 2, data.input.shape[1]))
            input_data[:data.input.shape[0], :] = data.input[:, :, 0]
            input_data[data.input.shape[0]:, :] = data.input[:, :, 1]
        # Decode
        X, Y, D = get_weight_data(data.target, input_data, model, sess, branch_factor=args.branch, beam_size=args.beam,
                               weight=[args.weight, 1 - args.weight], hash_length=args.hash,
                               gt_only=args.gt, history=args.history, features=args.features, min_diff=args.min_diff,
                               verbose=args.verbose, no_mlm=args.no_mlm)
    else:
        X = np.zeros((0, 0))
        Y = np.zeros(0)
        D = np.zeros(0)

        for file in glob.glob(os.path.join(args.MIDI, "*.mid")):
            if args.verbose:
                print(file)
            if args.step == "beat":
                data = dataMaps.DataMapsBeats()
                data.make_from_file(file,args.beat_gt,args.beat_subdiv,section, acoustic_model=args.acoustic)
            else:
                data = dataMaps.DataMaps()
                data.make_from_file(file, args.step, section=section, acoustic_model=args.acoustic)

            input_data = data.input
            if args.with_onsets:
                input_data = np.zeros((data.input.shape[0] * 2, data.input.shape[1]))
                input_data[:data.input.shape[0], :] = data.input[:, :, 0]
                input_data[data.input.shape[0]:, :] = data.input[:, :, 1]
            # Decode
            x, y, d = get_weight_data(data.target, input_data, model, sess, branch_factor=args.branch, beam_size=args.beam,
                                   weight=[[args.weight], [1 - args.weight]], hash_length=args.hash,
                                   gt_only=args.gt, history=args.history, features=args.features, min_diff=args.min_diff,
                                   verbose=args.verbose, no_mlm=args.no_mlm)

            if len(X) > 0:
                X = np.vstack((X, x))
            else:
                X = x

            Y = np.append(Y, y)
            D = np.append(D, d)

    print(X.shape)
    print(Y.shape)
    print(D.shape)

    if os.path.dirname(args.out) != '':
        os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Save data
    with gzip.open(args.out, "wb") as file:
        pickle.dump({'X' : X,
                     'Y' : Y,
                     'D' : D,
                     'history' : args.history,
                     'features' : args.features,
                     'no_mlm' : args.no_mlm}, file)
