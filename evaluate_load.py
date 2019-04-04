import numpy as np
import os
from eval_utils import compute_eval_metrics_frame, compute_eval_metrics_note
from mlm_training.utils import safe_mkdir
from dataMaps import DataMaps
import pickle


parser = argparse.ArgumentParser()
parser.add_argument('input_folder',type=str)
parser.add_argument('target_folder',type=str)
parser.add_argument("--step", type=str, choices=["time", "quant", "event"], help="Change the step type for frame timing. Either time (default), " +
                    "quant (for 16th notes), or event (for onsets).", default="time")
parser.add_argument('--with_offset', help="use offset for framewise metrics", action='store_true')

args = parser.parse_args()

input_folder = args.input_folder
target_folder = args.target_folder
# save_folder = 'results/baseline/raw'

frame = []
note = []

results = {}

# safe_mkdir(save_folder)

for fn in os.listdir(input_folder):
    if fn.endswith('.mid') and not fn.startswith('.'):
        filename_input = os.path.join(input_folder,fn)
        filename_target = os.path.join(target_folder,fn)
        print(filename)

        data = DataMaps()
        data.make_from_file(filename_target,'time',[0,30])

        input_roll = np.loadtxt(filename_input.replace('.mid','_pr.csv'))
        target_roll = data.target

        P_f,R_f,F_f = compute_eval_metrics_frame(baseline_roll,target_roll)
        P_n,R_n,F_n = compute_eval_metrics_note(baseline_roll,target_roll,min_dur=0.05,with_offset=args.with_offse)

        frame  += [[P_f,R_f,F_f]]
        note   += [[P_n,R_n,F_n]]

        results[fn] = [[P_f,R_f,F_f],[P_n,R_n,F_n]]
        # print([[P_f,R_f,F_f],[P_n,R_n,F_n]])


P_f, R_f, F_f = np.mean(frame, axis=0)
P_n, R_n, F_n = np.mean(note, axis=0)

print(f"Averages: Frame P,R,F: {P_f:.3f},{R_f:.3f},{F_f:.3f}, Note P,R,F: {P_n:.3f},{R_n:.3f},{F_n:.3f}")
sys.stdout.flush()

# pickle.dump(results,open(os.path.join('results/baseline/raw','results.p'), "wb"))