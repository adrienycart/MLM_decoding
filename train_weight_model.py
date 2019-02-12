import argparse
import numpy as np
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def train_model(X, Y, layers=None, weight=False):
    """
    Create and train a model from the given data.
    
    Parameters
    ==========
    X : np.ndarray
        An N x (history + num_features + 2) ndarray containing the N data points on which to train.
        
    Y : np.ndarray
        A length-N array, containing the targets for each data point.
        
    layers : list(int)
        The hidden layer sizes for the trained network. Defaults to None, which is logistic regression.
        
    weight : boolean
        True to have the model output prior weights, and False to have it output the prior directly.
        Defaults to False.
        
    Returns
    =======
    model : sklearn classifier
        A trained model.
    """
    if weight:
        convert_targets_to_weight(X, Y)
        
    if layers is None or len(layers) == 0:
        la = -1
        ac = -2
        strengths = strengths = np.abs(X[:, la] - X[:, ac])
        model = LogisticRegression().fit(X, Y, sample_weight=strengths)
    else:
        model = MLPClassifier(max_iter=1000, hidden_layer_sizes=layers).fit(X, Y)
        
    return model



        
def convert_targets_to_weight(X, Y):
    """
    Convert the given targets to train for a prior weight instead of a prior directly.
    
    Y will be edited in place.
    
    Parameters
    ==========
    X : np.ndarray
        An N x (history + num_features + 2) ndarray containing the N data points on which to train.
        
    Y : np.ndarray
        A length-N array, containing the targets for each data point. This will be edited in place.
    """
    la = -1
    ac = -2

    for i, x in enumerate(X):
        if Y[i] == 0:
            if x[la] < x[ac]:
                Y[i] = 1  #1 trains towards putting weight into the 2nd bin (which is the language weight
        else:
            if x[la] < x[ac]:
                Y[i] = 0
        



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--all", help="The data pickle file contains ALL possible data.", action="store_true")
    
    parser.add_argument("data", help="A pickle file containing a dictionary with X, Y, D, history, and features. " +
                        "This should be created by create_weight_data.py")
    
    parser.add_argument("out", help="The file to save the model to.")
    
    parser.add_argument("--min_diff", help="The minimum difference to use.", type=float, default=0.0)
    
    parser.add_argument("--history", help="The history length to use.", type=int, default=10)
    
    parser.add_argument("--layers", nargs='+', type=int, help="The hidden layer sizes of the network to train. " +
                        "If not given, logistic regression will be used.")
    
    parser.add_argument("--features", help="Use features (used if --all is given).", action="store_true")
    
    parser.add_argument("-w", "--weight", help="Create a model which outputs the prior weights (rather than " +
                        "the default, which will output the prior directly).", action="store_true")
    
    args = parser.parse_args()
    
    with open(args.data, "rb") as file:
        model_dict = pickle.load(file)
        
    X = model_dict['X']
    Y = model_dict['Y']
    D = model_dict['D']
    features = model_dict['features']
    history = model_dict['history']
    
    if args.all:
        if np.max(D) < min_diff:
            print("No training data found. Try with a lower min_diff.")
            sys.exit(0)
    
        data_points = np.where(D > args.min_diff)
        data_features = []
        
        if args.history > 0:
            data_features.extend(range(10 - args.history, 10))

        if args.features:
            data_features.extend(range(10, len(X[0]) - 2))

        data_features.append(-2)
        data_features.append(-1)

        X = X[data_points]
        X = X[:, data_features]
        Y = Y[data_points]
    
    model = train_model(X, Y, layers=args.layers, weight=args.weight)
    
    with open(args.out, "wb") as file:
        pickle.dump({'model' : model,
                     'history' : args.history if args.all else history,
                     'features' : args.features if args.all else features,
                     'weight' : args.weight}, file)
        