import skopt
import argparse
import os
import sys

import weight_search


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    args = parser.parse_args()
    
    print("Running for " + str(args.iters) + " iterations.")
    print("step type: " + args.step)
    print("saving output to " + args.output)
    print("using GPU " + args.gpu)
    sys.stdout.flush()
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    weight_search.set_step(args.step)
    weight_search.load_data()
    weight_search.load_model()
    
    dimensions = [[True, False], # GT
                  (0.0, 1.0), # min_diff
                  (0, 50) if args.step == "time" else (0, 10), # history
                  (0, 3), # num_layers
                  [True, False], # is_weight
                  [True, False]] # features

    opt = skopt.gp_minimize(weight_search.weight_search, dimensions, n_calls=10+args.iters, kappa=args.kappa, noise=0.0004, verbose=True, n_points=10)
    
    skopt.dump(opt, args.output)
