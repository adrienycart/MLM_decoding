import skopt
import numpy as np
import argparse
import os.path
import shutil

from optim_helper import get_filename


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("sko_file", help="The .sko file output by optimize_sk.py.")
    
    parser.add_argument("--dir", help="The directory where saved models are stored.",
                        required=True)
    
    parser.add_argument("--with_onsets", help="The saved blending model uses onsets.",
                        action="store_true")
    parser.add_argument("--no_mlm", help="The saved blending model uses no MLM data.",
                        action="store_true")
    parser.add_argument("--step", type=str, default='time',
                        choices=["time", "quant","quant_short", "event", "beat", "20ms"],
                        help="The step type used by the trained blending model.")
    
    parser.add_argument("-o", "--output", help="The filename to save the best model to, "
                        "if desired.")
    
    args = parser.parse_args()
    
    data = skopt.load(args.sko_file)
    
    best_index = np.argmin(data['func_vals'])
    params = data['x_iters'][best_index]
    
    print(f"Best F_n = {-data['func_vals'][best_index]} with params {params}")
    
    min_diff = params[0]
    history = int(params[1])
    num_layers = int(params[2])
    is_weight = params[3]
    features = params[4]
    
    filename = get_filename(min_diff, history, num_layers, features, args.no_mlm, args.with_onsets,
                            is_weight, args.step, num=0)
    
    print(f"Best model filename should be '{filename}'")
    
    print(f"Looking for {filename} in {args.dir}")
    
    full_path = os.path.join(args.dir, filename)
    
    if os.path.isfile(full_path):
        print("   Found!")
        
        if args.output is not None:
            print(f"Copying to {args.output}")
            shutil.copyfile(full_path, args.output)
    else:
        print("    Not found. Maybe arguments were incorrect?")