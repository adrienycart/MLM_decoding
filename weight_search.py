from dataMaps import DataMaps,convert_note_to_time, align_matrix
from eval_utils import compute_eval_metrics_frame, compute_eval_metrics_note
from mlm_training.model import Model, make_model_param
from mlm_training.utils import safe_mkdir
from decode import decode
from create_weight_data import get_weight_data
from train_weight_model import train_model

import os
import glob
import pickle
import warnings
import sys

import tensorflow as tf
import pretty_midi as pm
import numpy as np


step = {'step' : 'quant'}

def set_step(new_step):
    step['step'] = new_step
    
def weight_search(params, verbose=False):
    gt = params[0][0]
    min_diff = params[0][1]
    history = int(params[0][2])
    num_layers = int(params[0][3])
    is_weight = params[0][4]
    features = params[0][5]
    
    print(params)
    sys.stdout.flush()
    
    warnings.filterwarnings("ignore", message="tick should be an int.")
    folder = "data/outputs/valid"

    max_len = 30
    section = [0, max_len]
    
    note_range = [21,109]
    note_min = note_range[0]
    note_max = note_range[1]

    n_hidden = 256

    # Load model
    model_param = make_model_param()
    model_param['n_hidden'] = n_hidden
    model_param['n_steps'] = 1 # To generate 1 step at a time

    # Build model object
    model = Model(model_param)
    if step['step'] == "quant":
        model_path = "./ckpt/piano_midi/quant/quant_0.001_2/best_model.ckpt-374"
    elif step['step'] == "event":
        model_path = "./ckpt/piano_midi/quant/event_0.001/best_model.ckpt-284"
    elif step['step'] == "time":
        model_path = "./ckpt/piano_midi/quant/unquant_0.001/best_model.ckpt-463"
        
    sess,_ = model.load(model_path, model_path=model_path)

    # Get weight_model data
    if gt:
        with open("weight_models/data/gt.all." + step['step'] + ".pkl", "rb") as file:
            pkl = pickle.load(file)
    else:
        with open("weight_models/data/beam.all." + step['step'] + ".pkl", "rb") as file:
            pkl = pickle.load(file)
            
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
    data_features.append(-1)
    
    X = X[data_points]
    X = X[:, data_features]
    Y = Y[data_points]
    
    if len(X) == 0:
        print("No training data generated")
        sys.stdout.flush()
        return 0.0
    
    # Train weight model
    layers = []
    for i in range(num_layers):
        layers.append(5)

    weight_model = train_model(X, Y, layers=layers, weight=is_weight)
    
    weight_model_name = "weight_model."
    weight_model_name += "gt" if gt else "b10"
    weight_model_name += "_md" + str(min_diff)
    weight_model_name += "_h" + str(history)
    weight_model_name += "_l" + str(num_layers)
    if features:
        weight_model_name += "_f"
    weight_model_name += "_weight" if is_weight else "_prior"
    weight_model_name += "." + step['step'] + ".pkl"
    
    # Write out weight model
    with open("weight_models/models/" + weight_model_name, "wb") as file:
        pickle.dump({'model' : weight_model,
                     'history' : history,
                     'features' : features,
                     'weight' : is_weight}, file)

    results = {}
    frames = np.zeros((0, 3))
    notes = np.zeros((0, 3))

    for filename in glob.glob(os.path.join(folder, "*.mid")):
        print(filename)
        sys.stdout.flush()
        
        data = DataMaps()
        data.make_from_file(filename,step['step'],section)

        # Decode
        pr, priors, weights, combined_priors = decode(data.input, model, sess, branch_factor=50,
                            beam_size=200, union=False, weight=[[0.8], [0.2]],
                            out=None, hash_length=12, history=history, weight_model=weight_model,
                            verbose=verbose, features=features, is_weight=is_weight)

        if step['step'] is not "time":
            pr = convert_note_to_time(pr,data.corresp,max_len=max_len)

        data = DataMaps()
        data.make_from_file(filename, "time", section=section)
        target = data.target

        #Evaluate
        P_f,R_f,F_f = compute_eval_metrics_frame(pr,target)
        P_n,R_n,F_n = compute_eval_metrics_note(pr,target,min_dur=0.05)
        
        print(f"Frame P,R,F: {P_f:.3f},{R_f:.3f},{F_f:.3f}, Note P,R,F: {P_n:.3f},{R_n:.3f},{F_n:.3f}")
        sys.stdout.flush()

        frames = np.vstack((frames, [P_f, R_f, F_f]))
        notes = np.vstack((notes, [P_n, R_n, F_n]))

    P_f, R_f, F_f = np.mean(frames, axis=0)
    P_n, R_n, F_n = np.mean(notes, axis=0)
    
    print(f"Frame P,R,F: {P_f:.3f},{R_f:.3f},{F_f:.3f}, Note P,R,F: {P_n:.3f},{R_n:.3f},{F_n:.3f}")
    print(str(F_n) + ": " + str(params))
    sys.stdout.flush()
    return F_n