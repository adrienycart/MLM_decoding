import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../..')

from dataMaps import DataMaps,convert_note_to_time, align_matrix
from eval_utils import compute_eval_metrics_frame, compute_eval_metrics_note
from mlm_training.model import Model, make_model_param
from mlm_training.utils import safe_mkdir
from decode import decode, pad_x
from create_weight_data import get_weight_data
from train_weight_model import train_model

import glob
import pickle
import warnings
import gzip

import tensorflow as tf
import pretty_midi as pm
import numpy as np

model_dict = {'model' : None,
              'sess'  : None}

params = {'model_out' : None,
          'step'      : None,
          'acoustic'  : None}
    
data_dict = {'gt'    : None,
             'beam'  : None,
             'valid' : None}
    
def load_data_info(gt=None, beam=None, valid=None, model_path=None, n_hidden=256, step=None, model_out=".",
                   acoustic='kelz'):
    global params
    global data_dict
    global model_dict
    
    if gt is not None:
        with gzip.open(gt, "rb") as file:
            data_dict['gt'] = pickle.load(file)
            
    if beam is not None:
        with gzip.open(beam, "rb") as file:
            data_dict['beam'] = pickle.load(file)
            
    data_dict['valid'] = valid
    
    model_param = make_model_param()
    model_param['n_hidden'] = n_hidden
    model_param['n_steps'] = 1 # To generate 1 step at a time

    # Build model object
    model_dict['model'] = Model(model_param)
    model_dict['sess'],_ = model_dict['model'].load(model_path, model_path=model_path)
    
    params['step'] = step
    params['model_out'] = model_out
    params['acoustic'] = acoustic

    
most_recent_model = None

def get_most_recent_model():
    global most_recent_model
    return most_recent_model

    
def weight_search(params, num=0, verbose=False):
    print(params)
    sys.stdout.flush()
    
    gt = params[0]
    min_diff = params[1]
    history = int(params[2])
    num_layers = int(params[3])
    is_weight = params[4]
    features = params[5]
    
    history_context = 0
    prior_context = 0
    
    if len(params) > 6:
        history_context = params[6]
        prior_context = params[7]
        
    use_lstm = True
    if len(params) > 8:
        use_lstm = params[8]
    
    warnings.filterwarnings("ignore", message="tick should be an int.")

    max_len = 30
    section = [0, max_len]
    
    note_range = [21,109]
    note_min = note_range[0]
    note_max = note_range[1]

    # Load model
    model = model_dict['model']
    sess = model_dict['sess']

    # Get weight_model data
    pkl = data_dict['gt' if gt else 'beam']
            
    X = pkl['X']
    Y = pkl['Y']
    D = pkl['D']
    max_history = pkl['history']
    
    if np.max(D) < min_diff:
        print("No training data generated")
        sys.stdout.flush()
        return 0.0
    
    data_points = np.where(D > min_diff)
    data_features = []
    
    if history > 0:
        data_features.extend(range(max_history - history, max_history))
        
    if features:
        data_features.extend(range(max_history, len(X[0]) - 2))
    
    data_features.append(-2)
    
    if use_lstm:
        data_features.append(-1)
    
    X = X[:, data_features]
    
    if prior_context + history_context > 0:
        X_new = np.zeros((X.shape[0], X.shape[1] + prior_context * 4 + 2 * history_context * history))
        
        for i in range(int(X.shape[0] / 88)):
            x_frame = X[88 * i : 88 * (i + 1), :]
            
            X_new[88 * i : 88 * (i + 1), :] = pad_x(x_frame, x_frame[:, -2], x_frame[:, -1], x_frame[:, :history], history, history_context, prior_context)
            
        X = X_new
        
    X = X[data_points]
    Y = Y[data_points]
    
    if len(X) == 0:
        print("No training data generated")
        sys.stdout.flush()
        return 0.0
    
    # Train weight model
    print("Training weight model")
    sys.stdout.flush()
    layers = []
    for i in range(num_layers):
        layers.append(5)

    weight_model = train_model(X, Y, layers=layers, weight=is_weight)
    
    global most_recent_model
    most_recent_model = {'model' : weight_model,
                         'history' : history,
                         'features' : features,
                         'weight' : is_weight,
                         'history_context' : history_context,
                         'prior_context' : prior_context,
                         'use_lstm' : use_lstm}
    
    weight_model_name = "weight_model."
    weight_model_name += "gt" if gt else "b10"
    weight_model_name += "_md" + str(min_diff)
    weight_model_name += "_h" + str(history)
    weight_model_name += "_l" + str(num_layers)
    if features:
        weight_model_name += "_f"
    weight_model_name += "_hc" + str(history_context)
    weight_model_name += "_pc" + str(prior_context)
    if not use_lstm:
        weight_model_name += "_noLSTM"
    weight_model_name += "_weight" if is_weight else "_prior"
    weight_model_name += "." + step['step'] + "." + str(num) + ".pkl"
    
    # Write out weight model
    with open(os.path.join(params['model_out'], weight_model_name), "wb") as file:
        pickle.dump(most_recent_model, file)

    results = {}
    frames = np.zeros((0, 3))
    notes = np.zeros((0, 3))

    for filename in glob.glob(os.path.join(data_dir['valid'], "*.mid")):
        print(filename)
        sys.stdout.flush()
        
        data = DataMaps()
        data.make_from_file(filename,step['step'],section,acoustic_model=params['acoustic'])

        # Decode
        pr, priors, weights, combined_priors = decode(data.input, model, sess, branch_factor=5,
                            beam_size=50, weight=[[0.8], [0.2]],
                            out=None, hash_length=12, weight_model=weight_model,
                            verbose=verbose, weight_model_dict=most_recent_model)

        if params['step'] != "time":
            pr = convert_note_to_time(pr,data.corresp,data.input_fs,max_len=max_len)

        data = DataMaps()
        data.make_from_file(filename, "time", section=section, acoustic_model=params['acoustic'])
        target = data.target

        #Evaluate
        P_f,R_f,F_f = compute_eval_metrics_frame(pr,target)
        P_n,R_n,F_n = compute_eval_metrics_note(pr,target,min_dur=0.05)
        
        print(f"Frame P,R,F: {P_f:.3f},{R_f:.3f},{F_f:.3f}, Note P,R,F: {P_n:.3f},{R_n:.3f},{F_n:.3f}")
        sys.stdout.flush()

        frames = np.vstack((frames, [P_f, R_f, F_f]))
        notes = np.vstack((notes, [P_n, R_n, F_n]))
        
        if F_n < 0.05:
            print("Early stopping, F-measure too low.")
            sys.stdout.flush()
            return 0.0

    P_f, R_f, F_f = np.mean(frames, axis=0)
    P_n, R_n, F_n = np.mean(notes, axis=0)
    
    print(f"Frame P,R,F: {P_f:.3f},{R_f:.3f},{F_f:.3f}, Note P,R,F: {P_n:.3f},{R_n:.3f},{F_n:.3f}")
    print(str(F_n) + ": " + str(params))
    sys.stdout.flush()
    return -F_n
