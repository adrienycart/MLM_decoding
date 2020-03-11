"""Worker/helper functions that actually perform the Bayesian Optimization. These should not be
called directly. Rather, optimize_sk.py should be run from the command line."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../..')

from dataMaps import DataMaps,convert_note_to_time, align_matrix, DataMapsBeats
from eval_utils import (compute_eval_metrics_frame, compute_eval_metrics_note,
                        compute_eval_metrics_with_onset)
from mlm_training.model import Model, make_model_param
from mlm_training.utils import safe_mkdir
from decode import decode
from create_blending_data import add_noise_to_input_data
from train_blending_model import (train_model, convert_targets_to_weight, ablate,
                                  filter_data_by_min_diff, filter_X_features)

import glob
import pickle
import warnings
import gzip

import tensorflow as tf
import pretty_midi as pm
import numpy as np

# Global parameters
model_dict = {'model' : None,
              'sess'  : None}

global_params = {'model_out'  : None,
                 'step'       : None,
                 'acoustic'   : None,
                 'early_exit' : None,
                 'ablate'     : None}

data_dict = {'blending_data' : None,
             'valid'         : None,
             'noise'         : None,
             'noise_gauss'   : None}

def load_data_info(blending_data=None, valid=None, model_path=None, n_hidden=256, step=None,
                   beat_gt=None, beat_subdiv=None, model_out=".", acoustic='kelz',
                   early_exit=0.001, diagRNN=False, ablate_list=[], noise=None,
                   noise_gauss=False):
    """
    Set up the global parameter dictionaries to run Bayesian Optimization with the given
    settings. This is called by optimize_sk.py.
    """
    global global_params
    global data_dict
    global model_dict
    
    data_dict['valid'] = valid
    
    with gzip.open(blending_data, 'rb') as file:
        data_dict['blending_data'] = pickle.load(file)
        
    if noise is None:
        data_dict['noise'] = (data_dict['blending_data']['noise']
                              if 'noise' in data_dict['blending_data']
                              else None)
        data_dict['noise_gauss'] = (data_dict['blending_data']['noise_gauss']
                                    if 'noise_gauss' in data_dict['blending_data']
                                    else False)
    else:
        data_dict['noise'] = noise
        data_dict['noise_gauss'] = noise_gauss

    model_param = make_model_param()
    model_param['n_hidden'] = n_hidden
    model_param['n_steps'] = 1 # To generate 1 step at a time
    model_param['with_onsets'] = data_dict['blending_data']['with_onsets']
    if diagRNN:
        model_param['cell_type'] = "diagLSTM"

    # Build model object
    model_dict['model'] = Model(model_param)
    model_dict['sess'], _ = model_dict['model'].load(model_path, model_path=model_path)

    global_params['step'] = step
    global_params['beat_gt'] = beat_gt
    global_params['beat_subdiv'] = beat_subdiv
    global_params['model_out'] = model_out
    global_params['acoustic'] = acoustic
    global_params['early_exit'] = early_exit
    global_params['ablate'] = ablate_list
    
def get_filename(min_diff, history, num_layers, features, with_onsets,
                 is_weight, step, num=0):
    weight_model_name = "blending_model."
    weight_model_name += "md" + str(min_diff)
    weight_model_name += "_h" + str(history)
    weight_model_name += "_l" + str(num_layers)
    if features:
        weight_model_name += "_f"
    if with_onsets:
        weight_model_name += "_withOnsets"
    weight_model_name += "_weight" if is_weight else "_prior"
    weight_model_name += "." + step + "." + str(num) + ".pkl"
    return weight_model_name


most_recent_model = None

def get_most_recent_model():
    global most_recent_model
    return most_recent_model


def weight_search(params, num=0, verbose=False):
    global global_params
    print(params)
    sys.stdout.flush()

    # Parse params
    min_diff = params[0]
    history = int(params[1])
    num_layers = int(params[2])
    is_weight = params[3]
    features = params[4]

    warnings.filterwarnings("ignore", message="tick should be an int.")

    max_len = 30
    section = [0, max_len]

    # Load model
    model = model_dict['model']
    sess = model_dict['sess']

    # Get weight_model data
    pkl = data_dict['blending_data']

    X = pkl['X']
    Y = pkl['Y']
    D = pkl['D']
    max_history = pkl['history']
    features_available = pkl['features']
    with_onsets = pkl['with_onsets']

    # Filter data for min_diff
    X, Y = filter_data_by_min_diff(X, Y, np.maximum(D[:, 0], D[:, 1]) if with_onsets else D, min_diff)
    if len(X) == 0:
        print("No training data generated.")
        sys.stdout.flush()
        return 0.0

    # Filter X for desired input fields
    X = filter_X_features(X, history, max_history, features, features_available, with_onsets)
    
    # Ablate X
    X = ablate(X, global_params['ablate'], with_onsets=with_onsets)
    
    history = min(history, max_history)
    if features and not features_available:
        features = False

    # Train weight model
    print("Training weight model")
    sys.stdout.flush()
    layers = []
    for i in range(num_layers):
        layers.append(10 if with_onsets else 5)

    weight_model = train_model(X, Y, layers=layers, weight=is_weight, with_onsets=with_onsets)

    # Save model
    global most_recent_model
    most_recent_model = {'model' : weight_model,
                         'history' : history,
                         'features' : features,
                         'weight' : is_weight,
                         'with_onsets' : with_onsets,
                         'ablate' : global_params['ablate']}

    weight_model_name = get_filename(min_diff, history, num_layers, features, with_onsets, is_weight,
                                     global_params['step'])

    # Write out weight model
    with open(os.path.join(global_params['model_out'], weight_model_name), "wb") as file:
        pickle.dump(most_recent_model, file)

    # Evaluation
    results = {}
    frames = np.zeros((0, 3))
    notes = np.zeros((0, 3))

    for filename in sorted(glob.glob(os.path.join(data_dict['valid'], "*.mid"))):
        print(filename)
        sys.stdout.flush()

        if global_params['step'] == 'beat':
            data = DataMapsBeats()
            data.make_from_file(filename, global_params['beat_gt'], global_params['beat_subdiv'],
                                section,  acoustic_model=global_params['acoustic'], with_onsets=with_onsets)
        else:
            data = DataMaps()
            data.make_from_file(filename, global_params['step'], section,
                                acoustic_model=global_params['acoustic'], with_onsets=with_onsets)

        # Decode
        input_data = data.input
        if with_onsets:
            input_data = np.zeros((data.input.shape[0] * 2, data.input.shape[1]))
            input_data[:data.input.shape[0], :] = data.input[:, :, 0]
            input_data[data.input.shape[0]:, :] = data.input[:, :, 1]
            
        # Add noise
        input_data = add_noise_to_input_data(input_data, data_dict['noise'], data_dict['noise_gauss'])

        pr, priors, weights, combined_priors = decode(input_data, model, sess, branch_factor=5,
                        beam_size=50, weight=[[0.8], [0.2]], out=None, hash_length=12,
                        weight_model=weight_model, verbose=verbose, weight_model_dict=most_recent_model)
        
        # Evaluate
        if with_onsets:
            target_data = pm.PrettyMIDI(filename)
            corresp = data.corresp
            [P_f,R_f,F_f], [P_n,R_n,F_n], _, _ = compute_eval_metrics_with_onset(
                    pr, corresp, target_data, double_roll=True, min_dur=0.05, section=section
            )

        else:
            if global_params['step'] in ['quant','event','quant_short','beat']:
                pr = convert_note_to_time(pr, data.corresp, data.input_fs, max_len=max_len)
                
            data = DataMaps()
            if global_params['step'] == "20ms" or with_onsets:
                data.make_from_file(filename, "20ms", section=section, with_onsets=False, acoustic_model="kelz")
            else:
                data.make_from_file(filename, "time", section=section, with_onsets=False, acoustic_model="kelz")
            target = data.target

            #Evaluate
            P_f,R_f,F_f = compute_eval_metrics_frame(pr,target)
            P_n,R_n,F_n = compute_eval_metrics_note(pr,target, min_dur=0.05)

        print(f"Frame P,R,F: {P_f:.3f},{R_f:.3f},{F_f:.3f}, Note P,R,F: {P_n:.3f},{R_n:.3f},{F_n:.3f}")
        sys.stdout.flush()

        frames = np.vstack((frames, [P_f, R_f, F_f]))
        notes = np.vstack((notes, [P_n, R_n, F_n]))

        if F_n < global_params['early_exit']:
            print("Early stopping, F-measure too low.")
            sys.stdout.flush()
            return 0.0

    P_f, R_f, F_f = np.mean(frames, axis=0)
    P_n, R_n, F_n = np.mean(notes, axis=0)

    print(f"Frame P,R,F: {P_f:.3f},{R_f:.3f},{F_f:.3f}, Note P,R,F: {P_n:.3f},{R_n:.3f},{F_n:.3f}")
    print(str(F_n) + ": " + str(params))
    sys.stdout.flush()
    return -F_n
