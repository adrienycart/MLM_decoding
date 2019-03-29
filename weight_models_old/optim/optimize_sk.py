import skopt
import argparse
import os
import sys

import weight_search


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="The LSTM model to load, the best filename without extension.",
                        required=True)
    parser.add_argument("valid_data", help="The directory containing validation data files.")
    
    parser.add_argument("--acoustic", type=str, choices=["kelz", "bittner"], help="Change the acoustic model " +
                        "used in the files. Either kelz (default), or bittner.",
                        default="kelz")
    
    parser.add_argument("--step", type=str, choices=["time", "quant", "event"], help="Change the step type " +
                        "for frame timing. Either time (default), quant (for 16th notes), or event (for onsets).",
                        default="time")
    parser.add_argument("-o", "--output", help="The file to save the resulting optimization to. Defaults to optim.sko.",
                        default="optim.sko")
    parser.add_argument("--iters", help="The number of iterations to run optimization for (after the initial 10 " +
                        "random points). Defaults to 200.", type=int, default=200)
    parser.add_argument("--kappa", help="The kappa to use in the optimization. Defaults to 50.", type=float,
                        default=50)
    parser.add_argument("--gpu", help="The gpu to use. Defaults to 0.", default="0")
    parser.add_argument("--prior", help="Train for prior, rather than weight (default).", action="store_true")
    parser.add_argument("--beam_data", help="The data file to load beam data from. Defaults to None, which will " +
                        "not load any.", default=None)
    parser.add_argument("--gt_data", help="The data file to load ground truth data from. Defaults to None, which " +
                        "will not load any.", default=None)
    parser.add_argument("--model_dir", help="The directory to save model files to. Defaults to the current directory.",
                        default=".")
    args = parser.parse_args()
    
    print("Running for " + str(args.iters) + " iterations.")
    print("step type: " + args.step)
    print("saving output to " + args.output)
    print("using GPU " + args.gpu)
    print("Training for " + ("prior" if args.prior else "weight"))
    print("Loading LSTM " + args.model)
    print("Using gt data " + args.gt_data)
    print("Using beam data " + args.beam_data)
    print("Saving models to " + args.model_dir)
    sys.stdout.flush()
    
    os.makedirs(args.model_dir, exist_ok=True)
    
    if args.output is not None:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    weight_search.load_data_info(gt=args.gt_data, beam=args.beam_data, valid=args.valid_data, step=args.step,
                                 model_path=args.model, model_out=args.model_dir, acoustic=args.acoustic)
    
    dimensions = [[False], # GT
                  (0.1, 0.8), # min_diff
                  (5, 50) if args.step == "time" else (3, 10), # history
                  (1, 4), # num_layers
                  [not args.prior], # is_weight
                  [True, False], # features
                  [0], # history pitch context
                  [0], # prior context
                  [True]] # use LSTM

    opt = skopt.gp_minimize(weight_search.weight_search, dimensions, n_calls=10+args.iters,
                            kappa=args.kappa, noise=0.0004, verbose=True, n_points=10)
    
    skopt.dump(opt, args.output)
