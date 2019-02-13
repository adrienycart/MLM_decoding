import weight_search

import GPyOpt
import dill
import argparse
import sys


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=str, choices=["time", "quant", "event"], help="Change the step type " +
                        "for frame timing. Either time (default), quant (for 16th notes), or event (for onsets).",
                        default="time")
    parser.add_argument("-o", "--output", help="The directory to save outputs to. Defaults to optim.",
                        default="optim")
    parser.add_argument("--iters", help="The number of iterations to run optimization for. Defaults to 200.",
                        type=int, default=200)
    args = parser.parse_args()
    
    print("Running for " + str(args.iters) + " iterations.")
    print("step type: " + args.step)
    print("saving output to " + args.output)
    sys.stdout.flush()
    
    weight_search.set_step(args.step)
    
    domain = [{'name' : 'gt', 'type' : 'categorical', 'domain' : (True, False)},
              {'name' : 'min_diff', 'type' : 'continuous', 'domain' : (0, 1)},
              {'name' : 'history', 'type' : 'discrete', 'domain' : range(51) if args.step == "time" else range(11)},
              {'name' : 'num_layers', 'type' : 'discrete', 'domain' : range(4)},
              {'name' : 'is_weight', 'type' : 'categorical', 'domain' : (True, False)},
              {'name' : 'features', 'type' : 'categorical', 'domain' : (True, False)}]

    myBopt = GPyOpt.methods.BayesianOptimization(f=weight_search.weight_search, domain=domain, maximize=True,
                                                 verbosity=True, num_cores=4)

    myBopt.run_optimization(max_iter=args.iters, verbosity=True, report_file=args.output + "/Report.txt",
                            evaluations_file=args.output + "/Evaluations.txt", epsilon=0)

    print(myBopt.x_opt)
    print(myBopt.fx_opt)

    with open(args.output + "/optimized.dill", "wb") as file:
        dill.dump(myBopt, file)