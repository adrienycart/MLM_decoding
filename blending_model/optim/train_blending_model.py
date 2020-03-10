"""This file contains functions that can be used to train a blending model, as well as filter a
set of X, Y, D data (by both number of data points, and number of features). It can be run from
the command line."""
import argparse
import numpy as np
import pickle
import gzip

from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPClassifier


def train_model(X, Y, layers=None, weight=False, with_onsets=False):
    """
    Create and train a model from the given data.
    
    Parameters
    ==========
    X : np.ndarray
        An N x (history + num_features + 2) ndarray containing the N data points on which to train.
        
    Y : np.ndarray
        A length-N array, containing the targets for each data point. Or, an (N,2) target ndarray,
        if with_onsets is True.
        
    layers : list(int)
        The hidden layer sizes for the trained network. Defaults to None, which is logistic regression.
        
    weight : boolean
        True to have the model output prior weights, and False to have it output the prior directly.
        Defaults to False.
        
    with_onsets : boolean
        True to output presence and onset values. False for only presence.
        
    Returns
    =======
    model : sklearn classifier
        A trained model.
    """
    if weight:
        convert_targets_to_weight(X, Y, with_onsets=with_onsets)
        
    if layers is None or len(layers) == 0:
        la = -1
        ac = -2
        strengths = strengths = np.abs(X[:, la] - X[:, ac])
        regressor = MultiOutputRegressor(LogisticRegression()) if with_onsets else LogisticRegression()
        model = regressor.fit(X, Y, sample_weight=strengths)
    else:
        model = MLPClassifier(max_iter=1000, hidden_layer_sizes=layers).fit(X, Y)
        
    return model



        
def convert_targets_to_weight(X, Y, with_onsets=False):
    """
    Convert the given targets to train for a prior weight instead of a prior directly.
    
    Y will be edited in place.
    
    Parameters
    ==========
    X : np.ndarray
        An N x (history + num_features + 2) ndarray containing the N data points on which to train.
        
    Y : np.ndarray
        A length-N array, containing the targets for each data point. This will be edited in place.
        
    with_onsets : bool
        True if this data contains onsets. False otherwise.
    """
    if with_onsets:
        # Just treat each half as if it was without onsets
        convert_targets_to_weight(X[:, :X.shape[1] // 2], Y[:, 0], with_onsets=False)
        convert_targets_to_weight(X[:, X.shape[1] // 2:], Y[:, 1], with_onsets=False)
        Y[:, :] = 1 - Y # Convert because here, the output will be the acoustic weight for each
                        # target (presence, then onset), rather than the language weight in bin 0
                        # and the acoustic weight in bin 1.
        return
    
    la = -1
    ac = -2

    for i, x in enumerate(X):
        if Y[i] == 0:
            if x[la] < x[ac]:
                Y[i] = 1  #1 trains towards putting weight into the 2nd bin (which is the language weight)
        else:
            if x[la] < x[ac]:
                Y[i] = 0
                
                
def filter_data_by_min_diff(X, Y, D, min_diff):
    """
    Return the X and Y data containing only those points whose distance is greater than
    or equal to the given min distance.
    
    Parameters
    ----------
    X : np.ndarray
        (N, ?) array conatining the N data point inputs.
        
    Y : np.ndarray or np.array
        (N,) array or (N, ?) array containing the labels.
        
    D : np.array
        (N,) array containing the difference for each data point.
        
    min_diff : float
        The minimum difference to return a data point.
        
    Returns
    -------
    X : np.ndarray
        The given X data points, minus those points whose difference is less than the given min_diff.
        
    Y : np.ndarray or np.array
        The given Y data points, minus those points whose difference is less than the given min_diff.
    """
    data_points = np.where(D > min_diff)
    return X[data_points], Y[data_points]


def filter_X_features(X, history, max_history, features, features_available, with_onsets):
    """
    Filter the given X data points down to the desired input features.
    
    Parameters
    ----------
    X : np.ndarray
        (N, ?) array, the N data points with all possible features.
        
    history : int
        The desired history length for each data point to retain. If this is greater than max_history,
        it is set to max_history.
        
    max_history : int
        The maximum history length that each data point currently has.
        
    features : bool
        True to use features. False otherwise. If features_available is False, this is set to False.
        
    features_available : bool
        True if features are in the input data. False otherwise.
        
    with_onsets : bool
        True if the given data contains onsets. False otherwise.
    """
    if max_history < history:
        print(f"The amount of history in the data ({max_history}) is less than the desired history ({history})."
              f" Returning data with history ({max_history}).")
        history = max_history
        
    if features and not features_available:
        print("Requested features but they are not in the data. Returning without features.")
        features = False
        
    data_features = []

    if history > 0:
        data_features.extend(range(max_history - history, max_history))

    if features:
        data_features.extend(range(max_history, len(X[0]) // 2 - 2 if with_onsets else len(X[0]) - 2))

    data_features.append(-2)
    data_features.append(-1)

    if with_onsets:
        X_presence, X_onsets = np.split(X, 2, axis=1)
        X = np.hstack((X_presence[:, data_features], X_onsets[:, data_features]))
    else:
        X = X[:, data_features]
        
    return X



def ablate(X, ablation, with_onsets=False):
    """
    Filter the features indicated by the ablation list from the X data points by setting them all to 0.
    
    Parameters
    ----------
    X : np.ndarray
        An (N,(num_features)) array of N input data points.
        
    ablation : list
        A list of the indexes of the columns of X to set to 0.
        
    with_onsets : boolean
        If true, split the X array in half and zero out each half according to the ablate list.
    
    Returns
    -------
    X : np.ndarray
        The input array, but with the corresponding features set to 0.
    """
    if len(ablation) > 0:
        if with_onsets:
            X_presence, X_onsets = np.split(X, 2, axis=1)
            X_presence = ablate(X_presence, ablation, with_onsets=False)
            X_onsets = ablate(X_onsets, ablation, with_onsets=False)
            X = np.hstack((X_presence, X_onsets))
        else:
            X[:, ablation] = 0
    return X


def add_acoustic_noise(X, noise, noise_gauss=False, with_onsets=False):
    """
    Add some noise to the acoustic samples in the given X data points.
    
    Parameters
    ----------
    X : np.ndarray
        An (N,(num_features)) array of N input data points.
        
    noise : float
        The amount of noise to add. None for no noise.
        
    noise_gauss : boolean
        The type of noise to add. If False, uniform noise on the range (-noise, noise).
        If True, Gaussian noise with standard deviation noise.
        
    with_onsets : boolean
        If true, split the X array in half and add noise to each half.
        
    Returns
    -------
    X : np.ndarray
        The input array, but with the given noise added.
    """
    if noise is not None and abs(noise) > 0:
        if with_onsets:
            X_presence, X_onsets = np.split(X, 2, axis=1)
            X_presence = add_acoustic_noise(X_presence, noise, noise_gauss=noise_gauss, with_onsets=False)
            X_onsets = add_acoustic_noise(X_onsets, noise, noise_gauss=noise_gauss, with_onsets=False)
            X = np.hstack((X_presence, X_onsets))
        else:
            noise = abs(noise)
            noise_func = np.random.randn if noise_gauss else np.random.rand
            X[:, -2] = np.clip(X[:, -2] + noise * noise_func(len(X)), 0, 1)
    return X



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("data", help="A pickle file containing a dictionary with X, Y, D and other data. " +
                        "This should be created by create_blending_data.py")
    
    parser.add_argument("out", help="The file to save the model to.")
    
    parser.add_argument("--min_diff", help="The minimum difference to use.", type=float, default=0.0)
    
    parser.add_argument("--history", help="The history length to use.", type=int, default=10)
    
    parser.add_argument("--layers", nargs='+', type=int, help="The hidden layer sizes of the network to train. " +
                        "If not given, logistic regression will be used.")
    
    parser.add_argument("--no_features", help="Don't use features.", action="store_true")
    
    parser.add_argument("--ablate", help="Indexes to ablate (set to 0) from the input. Important indices are:\n"
                        "\t\t-11, -10 = acoustic, language uncertainty\n"
                        "\t\t-9, -8   = acoustic, language entropy\n"
                        "\t\t-7, -6   = acoustic, language mean\n"
                        "\t\t-5, -4   = acoustic, language flux\n"
                        "\t\t-3       = pitch"
                        "\t\t-2, -1   = acoustic, language prior",
                        nargs='+', type=int, default=[])
    parser.add_argument("--no_mlm", help="Suppress all MLM inputs. Shortcut for --ablate -10 -8 -6 -4 -1",
                        action="store_true")
    
    parser.add_argument("--noise", help="Add uniform random noise to the acoustic model's activations, "
                        "on the range (-noise, noise), or Gaussian noise with standard deviation noise, "
                        "if --noise_gauss is also given.", type=float, default=None)
    parser.add_argument("--noise_gauss", help="Make the added noise Gaussian with the given arg being the "
                        "standard deviation of the desired noise.", action="store_true")
    
    parser.add_argument("-w", "--weight", help="Create a model which outputs the prior weights (rather than " +
                        "the default, which will output the prior directly).", action="store_true")
    
    args = parser.parse_args()
    
    if args.no_mlm:
        for index in [-10, -8, -6, -4, -1]:
            if index not in args.ablate:
                args.ablate.append(index)
    
    with gzip.open(args.data, "rb") as file:
        model_dict = pickle.load(file)
        
    # Read from model data
    X = model_dict['X']
    Y = model_dict['Y']
    D = model_dict['D']
    features_available = model_dict['features']
    max_history = model_dict['history']
    with_onsets = model_dict['with_onsets']
    
    # Filter for min_diff
    X, Y = filter_data_by_min_diff(X, Y, np.maximum(D[:, 0], D[:, 1]) if with_onsets else D, args.min_diff)
    if len(X) == 0:
        print("No training data found. Try with a lower min_diff.")
        sys.exit(0)

    # Filter X for desired input fields
    X = filter_X_features(X, args.history, max_history, not args.no_features, features_available, with_onsets)
    
    X = ablate(X, args.ablate, with_onsets=with_onsets)
    
    X = add_acoustic_noise(X, args.noise, noise_gauss=args.noise_gauss, with_onsets=with_onsets)
    
    model = train_model(X, Y, layers=args.layers, weight=args.weight, with_onsets=with_onsets)
    
    with open(args.out, "wb") as file:
        pickle.dump({'model' : model,
                     'history' : args.history,
                     'features' : not args.no_features,
                     'weight' : args.weight,
                     'with_onsets' : with_onsets,
                     'ablate' : args.ablate}, file)
        