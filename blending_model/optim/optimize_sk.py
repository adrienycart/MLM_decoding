"""This file performes the Bayesian Optimization search for a blending model, using the functions
in optim_helper.py."""
import skopt
import argparse
import os
import sys
import numpy as np

from skopt.callbacks import CheckpointSaver, EarlyStopper

from early_stopper import EarlyStopperNoImprovement
import optim_helper


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="The LSTM model to load, the best filename without extension.",
                        required=True)
    parser.add_argument("valid_data", help="The directory containing validation data files.")
    parser.add_argument("--blending_data", help="The data file to load blending data from. This should've "
                        "been created using create_blending_data.py", required=True)

    parser.add_argument("--acoustic", type=str, choices=["kelz", "bittner"], help="Change the acoustic model " +
                        "used in the files. Either kelz (default), or bittner.",
                        default="kelz")

    parser.add_argument("--step", type=str, choices=["time", "quant","quant_short", "event", "beat", "20ms"], help="Change the step type " +
                        "for frame timing. Either time (default), quant (for 16th notes), or event (for onsets).",
                        default="time")
    parser.add_argument('--beat_gt',action='store_true',help="with beat timesteps, use ground-truth beat positions")
    parser.add_argument('--beat_subdiv',type=str,help="with beat timesteps, beat subdivisions to use (comma separated list, without brackets)",default='0,1/4,1/3,1/2,2/3,3/4')

    parser.add_argument("-o", "--output", help="The file to save the resulting optimization to. Defaults to optim.sko.",
                        default="optim.sko")
    parser.add_argument("--iters", help="The number of iterations to run optimization for (after the initial 10 " +
                        "random points). Defaults to 200.", type=int, default=200)
    parser.add_argument("--kappa", help="The kappa to use in the optimization. Defaults to 50.", type=float,
                        default=50)
    parser.add_argument("--gpu", help="The gpu to use. Defaults to 0.", default="0")
    parser.add_argument("--cpu", help="Use CPU.", action="store_true")
    parser.add_argument("--prior", help="Train for prior, rather than weight (default).", action="store_true")
    parser.add_argument("--model_dir", help="The directory to save model files to. Defaults to the current directory.",
                        default=".")
    parser.add_argument("--give_up", help="Tell the model to quit the computation of a point if the value of any " +
                        "piece's notewise F-measure is below this amount. Defaults to 0.001.", type=float,
                        default=0.001)
    parser.add_argument("--early_stopping", help="Stop optimization if the best result was not for this number "
                        "of iterations.", type=int, default=50)
    parser.add_argument("--diagRNN", help="Use diagonal RNN units", action="store_true")
    parser.add_argument("--load", help="Continue optimization from the file in --output.", action="store_true")
    parser.add_argument("--ablate", help="Indexes to ablate (set to 0) from the input. Important indices are:\n"
                        "\t\t-11, -10 = acoustic, language uncertainty\n"
                        "\t\t-9, -8   = acoustic, language entropy\n"
                        "\t\t-7, -6   = acoustic, language mean\n"
                        "\t\t-5, -4   = acoustic, language flux\n"
                        "\t\t-3       = pitch"
                        "\t\t-2, -1   = acoustic, language prior",
                        nargs='+', type=int, default=[])
    parser.add_argument("--no_mlm", help="Suppress all MLM inputs. Shortcut for --ablate -10 -8 -6 -4 -1",
                        action="store_true")
    parser.add_argument("--no_features", help="Suppress all features. Shortcut for --ablate -11 -10 -9 -8 "
                        "-7 -6 -5 -4 -3", action="store_true")
    parser.add_argument("--no_history", help="Set history to 0 for all models.", action="store_true")

    args = parser.parse_args()
    
    if args.no_mlm:
        for index in [-10, -8, -6, -4, -1]:
            if index not in args.ablate:
                args.ablate.append(index)
                
    if args.no_features:
        for index in [-11, -10, -9, -8, -7, -6, -5, -4, -3]:
            if index not in args.ablate:
                args.ablate.append(index)

    print("Running for at most " + str(args.iters) + " iterations.")
    print(f"Stopping if no improvement for at least {args.early_stopping} iterations.")
    print("step type: " + args.step)
    print("Ablating: " + str(args.ablate))
    print("saving output to " + args.output)
    if args.cpu:
        print("Using CPU")
    else:
        print("using GPU " + args.gpu)
    print("Training for " + ("prior" if args.prior else "weight"))
    print("Loading LSTM " + args.model)
    print("Using blending data " + args.blending_data)
    print("Early exit threshold at " + str(args.give_up))
    print("Saving models to " + args.model_dir)
    sys.stdout.flush()

    os.makedirs(args.model_dir, exist_ok=True)

    if args.output is not None and os.path.dirname(args.output) != '':
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if args.cpu:
        args.gpu = ""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    optim_helper.load_data_info(blending_data=args.blending_data, valid=args.valid_data, step=args.step,
                                model_path=args.model, model_out=args.model_dir, acoustic=args.acoustic,
                                early_exit=args.give_up,diagRNN=args.diagRNN, beat_gt=args.beat_gt,
                                beat_subdiv=args.beat_subdiv, ablate_list=args.ablate)

    if args.step in ["time", "20ms"]:
        history = (5, 50)
    elif args.step in ["beat"]:
        history = (4, 12)
    else:
        history = (3, 10)
        
    if args.no_history:
        history = [0]
    
    dimensions = [(0.1, 0.8), # min_diff
                  history, # history
                  (1, 4), # num_layers
                  [not args.prior], # is_weight
                  [True]] # features

    callbacks = []
    callbacks.append(CheckpointSaver(args.output))
    callbacks.append(EarlyStopperNoImprovement(args.early_stopping))

    x0 = None
    y0 = None
    try:
        if args.load:
            res = skopt.load(args.output)
            x0 = res.x_iters
            y0 = res.func_vals
    except:
        print("Could not load from checkpoint " + args.output)
        print("Starting fresh")
    
    n_random_starts = max(10 - (len(x0) if x0 is not None else 0), 0)
    n_calls = max(args.iters - (len(x0) if x0 is not None else 0), 0)

    opt = skopt.gp_minimize(optim_helper.weight_search, dimensions, n_calls=n_calls,
                            kappa=args.kappa, noise=0.0004, verbose=True, n_points=10,
                            x0=x0, y0=y0, callback=callbacks,
                            n_random_starts=n_random_starts)

    skopt.dump(opt, args.output)
