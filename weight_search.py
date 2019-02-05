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

import tensorflow as tf
import pretty_midi as pm
import numpy as np

def weight_search(params):
    gt = params[0][0]
    min_diff = params[0][1]
    history = int(params[0][2])
    num_layers = int(params[0][3])
    is_weight = params[0][4]
    features = params[0][5]
    
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
    sess,_ = model.load("./ckpt/piano_midi/quant/quant_0.001_2/best_model.ckpt-374",
                        model_path="./ckpt/piano_midi/quant/quant_0.001_2/best_model.ckpt-374")

    # Get weight_model data
    X = None
    Y = np.zeros(0)

    for file in glob.glob(os.path.join(folder, "*.mid")):
        data = DataMaps()
        data.make_from_file(file, "quant", section=section)

        # Decode
        x, y = get_weight_data(data.target, data.input, model, sess, branch_factor=1 if gt else 10, beam_size=1 if gt else 10,
                               union=False, weight=[[0.8], [0.2]], hash_length=12,
                               gt_only=gt, history=history, features=features, min_diff=min_diff, verbose=False)
        
        if X is not None:
            X = np.vstack((X, x))
        else:
            X = x
        Y = np.append(Y, y)
            
    # Train weight model
    layers = []
    for i in range(num_layers):
        layers.append(5)

    weight_model = train_model(X, Y, layers=layers, weight=is_weight)

    results = {}
    frames = np.zeros((0, 3))
    notes = np.zeros((0, 3))

    for filename in glob.glob(os.path.join(folder, "*.mid")):
        data = DataMaps()
        data.make_from_file(filename,"quant",section)

        # Decode
        pr, priors, weights, combined_priors = decode(data.input, model, sess, branch_factor=50,
                            beam_size=200, union=False, weight=[[0.8], [0.2]],
                            out=None, hash_length=12, history=history, weight_model=weight_model,
                            verbose=False, features=features, is_weight=is_weight)

        pr = convert_note_to_time(pr,data.corresp,max_len=max_len)

        data = DataMaps()
        data.make_from_file(filename, "time", section=section)
        target = data.target

        #Evaluate
        P_f,R_f,F_f = compute_eval_metrics_frame(pr,target)
        P_n,R_n,F_n = compute_eval_metrics_note(pr,target,min_dur=0.05)

        frames = np.vstack((frames, [P_f, R_f, F_f]))
        notes = np.vstack((notes, [P_n, R_n, F_n]))

    P_f, R_f, F_f = np.mean(frames, axis=0)
    P_n, R_n, F_n = np.mean(notes, axis=0)
    
    print(str(F_n) + ": " + str(params))
    return F_n