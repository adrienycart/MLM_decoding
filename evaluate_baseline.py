import numpy as np
import os
from eval_utils import compute_eval_metrics_frame, compute_eval_metrics_note
from mlm_training.utils import safe_mkdir
from dataMaps import DataMaps
import pickle


baseline_folder = 'data/outputs_default_config_split/test'
save_folder = 'results/baseline/raw'

frame = []
note = []

results = {}

safe_mkdir(save_folder)

for fn in os.listdir(baseline_folder):
    if fn.endswith('.mid') and not fn.startswith('.'):
        filename = os.path.join(baseline_folder,fn)
        print(filename)

        data = DataMaps()
        data.make_from_file(filename,'time',[0,30])

        baseline_roll = (data.input>0.5).astype(int)
        target_roll = data.target

        P_f,R_f,F_f = compute_eval_metrics_frame(baseline_roll,target_roll)
        P_n,R_n,F_n = compute_eval_metrics_note(baseline_roll,target_roll,min_dur=0.05)

        frame  += [[P_f,R_f,F_f]]
        note   += [[P_n,R_n,F_n]]

        results[fn] = [[P_f,R_f,F_f],[P_n,R_n,F_n]]
        # print([[P_f,R_f,F_f],[P_n,R_n,F_n]])




print(np.mean(np.array(frame),axis=0))
print(np.mean(np.array(note),axis=0))

pickle.dump(results,open(os.path.join('results/baseline/raw','results.p'), "wb"))
