import tensorflow as tf
from tensorflow import keras
import numpy as np
import pickle
import argparse
import datetime
import gzip
import os

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../..')

import decode


def load_data_from_pkl_file(pkl_file, min_diff, history, ac_pitch_window, la_pitch_window,
                            features, is_weight):
    """
    Load data points from the given gzipped pickle file with the given settings.
    
    Parameters
    ==========
    pkl_file : string
        The filename of the gzipped pickle file containing the data to load. The pickle file
        should contain a dictionary with the following fields:
        
            history: an integer, the history length that is saved with each data point.
            beam: an integer, the beam size used to create the data points in X.
            features: boolean, whether features are included with each data point.
            A: num_pieces x 88 x num_frames array containing the acoustic prior for each frame
                and pitch. Note that num_frames is different for each piece.
            X: (88 * num_frames_total * beam) x (history + num_features + 1) array,
                where N is the number of data points, and each data point contains the history
                of samples, followed by features (if features is True), followed by the
                language model's prior for the corresponding frame. The data points are ordered
                by piece, frame, beam, and then pitch.
            Y: (88 * num_frames_total * beam)-length array, containing the ground truth
                piano roll for the corresponding frame.
            D: (88 * num_frames_total * beam)-length array, containing the difference
                between the language prior and the acoustic prior for each data point.
            
    min_diff : float
        The minimum difference to load and use a data point. This will be compared to D from
        the pickle dictionary.
        
    history : int
        The history length to use in each loaded data point.
        
    ac_pitch_window : list(int)
        The pitches around each data point to use, for its acoustic history.
        
    la_pitch_window : list(int)
        The pitches around each data point to use, for its sample history.
        
    features : boolean
        Whether to use features (True) or not (False).
        
    is_weight : boolean
        Whether to create targets for priors (False, 1 for acoustic, 0 for language), or for
        weights directly (True).
        
    Returns
    =======
    X : np.ndarray
        The requested X data points.
        
    Y : np.array
        The requested Y data points.
        
    D : np.array
        The difference associated with each returned data point.
    """
    with gzip.open(pkl_file, "rb") as file:
        pkl = pickle.load(file)
        
    if history > pkl['history']:
        raise Exception("Desired history length (" + str(history) + ") greater than that saved in data file (" +
                        str(pkl['history']) + ").")
        
    if features and not pkl['features']:
        raise Exception("Wanted to use features, but none are saved in data file.")
    
    used_D = np.where(pkl['D'] >= min_diff)[0]
    
    X = []
    Y = pkl['Y'][used_D]
        
    # Doing it in loops because it's easier. If it takes a prohibitavely long time this could be vectorized.
    frame_num_total = -1
    base_index = -88
    for piece_num, acoustic in enumerate(pkl['A']):
        acoustic = np.transpose(acoustic)
        
        for frame_num, frame in enumerate(np.transpose(acoustic)):
            frame_num_total += 1
            
            for beam in range(1 if frame_num == 0 else pkl['beam']):
                base_index += 88
                language = pkl['X'][base_index:base_index + 88, :pkl['history']]
                
                X.extend(decode.get_data_tf(history, ac_pitch_window, la_pitch_window, acoustic, language,
                                            frame_num, np.where(pkl['D'][base_index:base_index+88] >= min_diff)[0]))
    
    X = np.hstack((np.array(X), pkl['X'][used_D, (pkl['history'] if features else -2):]))
    
    if is_weight:
        convert_targets_to_weight(X, Y)
    
    return X, Y, pkl['D'][used_D]



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
                Y[i] = 1 #1 trains towards putting weight into the 2nd bin (which is the language weight)
        else:
            if x[la] < x[ac]:
                Y[i] = 0




def get_num_features(size, history, ac_pitch_window_size, la_pitch_window_size):
    """
    Get the number of features in an X matrix with the given size.
    
    Parameters
    ==========
    size : int
        The length of each data point.
        
    history : int
        The history length.

    ac_pitch_window_size : int
        The size of the acoustic pitch window.

    la_pitch_window_size : int
        The size of the sample pitch window.
        
    Returns
    =======
    num_features : int
        The number of features contained in the X matrix.
    """
    return size - history * (ac_pitch_window_size + la_pitch_window_size)



def make_model(history, ac_pitch_window_size, la_pitch_window_size, num_features, ac_num_pitch_convs=5,
               ac_num_history_convs=10, la_num_convs=5):
    """
    Create and return a new keras model.

    Parameters
    ==========
    history : int
        The history length we will use in this model.

    ac_pitch_window_size : int
        The size of the acoustic pitch window in the data we will be given.

    la_pitch_window_size : int
        The size of the sample pitch window in the data we will be given.

    num_features : int
        The number of features we will be given.

    ac_num_pitch_convs : int
        The number of channels to use when convolving across the acoustic pitches.
        Defaults to 5.

    ac_num_history_convs : int
        The number of channels to use when convolving across the acoustic history dimension.
        Defaults to 10.

    la_num_convs : int
        The number of channels to use when convolving across the samples.
        Defaults to 5.
    """
    acoustic_in = keras.layers.Input(shape=(ac_pitch_window_size, history,), dtype='float', name='acoustic')
    language_in = keras.layers.Input(shape=(la_pitch_window_size, history,), dtype='float', name='language')
    features_in = keras.layers.Input(shape=(num_features,), dtype='float', name='features')

    # Acoustic model history - 1d convolutions in each direction
    acoustic = keras.layers.Reshape((ac_pitch_window_size, history, 1),
                                    input_shape=(ac_pitch_window_size * history,))(acoustic_in)
    acoustic = keras.layers.Conv2D(ac_num_pitch_convs, (ac_pitch_window_size, 1))(acoustic)
    acoustic = keras.layers.Permute((3, 2, 1))(acoustic)
    acoustic = keras.layers.Conv2D(ac_num_history_convs, (ac_num_pitch_convs, min(history, 5)))(acoustic)
    acoustic = keras.layers.Flatten()(acoustic)

    # Language model history - series of 2D convolutions
    language = keras.layers.Reshape((la_pitch_window_size, history, 1),
                                    input_shape=(la_pitch_window_size * history,))(language_in)
    language = keras.layers.Conv2D(la_num_convs, (3, 3), strides=1)(language)
    language = keras.layers.Conv2D(la_num_convs, (3, 3), strides=2)(language)
    language = keras.layers.Flatten()(language)

    # Dense layers
    dense_in = keras.layers.Concatenate()([acoustic, language, features_in])

    dense = keras.layers.Dense(20, activation='relu')(dense_in)
    dense = keras.layers.Dropout(0.2)(dense)

    dense = keras.layers.Dense(20, activation='relu')(dense)
    dense = keras.layers.Dropout(0.2)(dense)

    dense = keras.layers.Dense(20, activation='relu')(dense)
    dense = keras.layers.Dropout(0.2)(dense)

    dense_out = keras.layers.Dense(1, activation='sigmoid')(dense)

    return keras.models.Model(inputs=[acoustic_in, language_in, features_in], outputs=dense_out)



def train_model(model, X, Y, sample_weight=None, optimizer='adam', epochs=100, checkpoints=[]):
    """
    Train a keras model.
    
    Parameters
    ==========
    model : tf.keras.models.Model
        The keras model to train.
        
    X : np.ndarray
        The input data points, as [acoustic_in, language_in, features_in].
        
    Y : np.array
        The targets.
        
    sample_weight : np.array
        A weight to give to each sample. Defaults to None (uniform weighting).
        
    optimizer : string OR tf.keras.optimizers.Optimizer
        The optimizer to use when training. Defaults to 'adam', the default-settings Adam optimizer.
        
    epochs : int
        The number of epochs to train for. Defaults to 100.
        
    checkpoints : list(keras.callbacks.ModelCheckpoint)
        A list of checkpoints which will save the model.
    """
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['mse', 'accuracy'])
    
    model.fit(X, Y, sample_weight=sample_weight, validation_split=0.1, epochs=epochs, batch_size=32,
              callbacks=checkpoints)
    
    
    
    
def train_model_full(data_file, history=5, ac_pitch_window=[-19, -12, 0, 12, 19],
                     la_pitch_window=list(range(-12, 13)), min_diff=0.0, features=True,
                     out="ckpt", is_weight=True, epochs=100, no_lstm=False):
    """
    Train a model fully given some data and parameters.
    
    Parameters
    ==========
    data_file : string
        The filename of a gzipped pickle data file.
        
    history : int
        The history length to use in each loaded data point. If it is greater than the history
        value from the pickle dictionary, that one is used instead. Defaults to 5.
        
    ac_pitch_window : list(int)
        The pitches around each data point to use, for its acoustic history. Defaults to [-19, -12, 0, 12, 19].
        
    la_pitch_window : list(int)
        The pitches around each data point to use, for its sample history. Defaults to list(range(-12,13)).
        
    min_diff : float
        The minimum difference to load and use a data point. This will be compared to D from
        the pickle dictionary. Defaults to 0.0.
        
    features : boolean
        Whether to use features (True) or not (False). If this is True, but the value of features
        in the given pickle dictionary is False, False is used instead. Defaults to True.
        
    is_weight : boolean
        Whether to create targets for priors (False, 1 for acoustic, 0 for language), or for
        weights directly (True). Defaults to True.
        
    epochs : int
        The number of epochs to train the model for. Defaults to 100.
        
    out : string
        The directory to save the checkpoints to. Defaults to ckpt.
        
    no_lstm : boolean
        Flage to include (False) or not include (True) the LSTM prior in the data points. Defaults to False.
    """
    print("Loading data...")
    X, Y, D = load_data_from_pkl_file(data_file, min_diff, history, ac_pitch_window, la_pitch_window,
                                      features, is_weight)
    
    if no_lstm:
        X = X[:, :-1]
    
    # Shuffle so that validation set is random 10%
    shuffle = list(range(X.shape[0]))
    np.random.shuffle(shuffle)
    X = X[shuffle]
    Y = Y[shuffle]
    D = D[shuffle]
    
    # Split into separate data sections
    ac_pitch_window_size = len(ac_pitch_window)
    la_pitch_window_size = len(la_pitch_window)
    num_features = get_num_features(X.shape[1], history, ac_pitch_window_size, la_pitch_window_size)
    
    acoustic_in = X[:, :len(ac_pitch_window) * history].reshape(-1, len(ac_pitch_window), history)
    language_in = X[:, len(ac_pitch_window) * history:history * (len(ac_pitch_window) + len(la_pitch_window))].reshape(-1, len(la_pitch_window), history)
    features_in = X[:, history * (len(ac_pitch_window) + len(la_pitch_window)):]
    
    print("Loaded " + str(X.shape[0]) + " data points of size " + str(X.shape[1]) + ".")
    print("Training model...")
    
    model = make_model(history, ac_pitch_window_size, la_pitch_window_size, num_features,
                       ac_num_pitch_convs=5, ac_num_history_convs=10, la_num_convs=5)
    
    checkpoint = keras.callbacks.ModelCheckpoint(os.path.join(out, 'model.{epoch:03d}-{loss:.4f}.ckpt'))
    checkpoint_best = keras.callbacks.ModelCheckpoint(os.path.join(out, 'best.ckpt'), monitor='val_loss',
                                                      save_best_only=True)
    checkpoints = [checkpoint_best, checkpoint]
    
    train_model(model, [acoustic_in, language_in, features_in], Y, sample_weight=D, epochs=epochs,
                checkpoints=checkpoints)
    
    # Reload to save model in easily-loadable format
    model.load_weights(os.path.join(out, 'best.ckpt'))
    model.save(os.path.join(out, 'best.h5'))
    
    with open(os.path.join(out, 'dict.pkl'), "wb") as file:
        pickle.dump({'model_path' : os.path.join(out, 'best.h5'),
                     'history' : history,
                     'ac_pitch_window' : ac_pitch_window,
                     'la_pitch_window' : la_pitch_window,
                     'features' : features,
                     'is_weight' : is_weight,
                     'no_lstm' : no_lstm},
                    file)
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("data", help="The gzipped pickle file to load data from.")
    
    parser.add_argument("--history", help="The history length to use. Defaults to 5.",
                        type=int, default=5)
    
    parser.add_argument("-A", "--a_pitches", nargs='+', type=int, help="The pitches to save data from for the acoustic " +
                        "model, relative " +
                        "to the current pitch. 0 will automatically be appended to this list.", default=[])
    
    parser.add_argument("-L", "--l_pitches", nargs='+', type=int, help="The pitches to save data from for the language " +
                        "model, relative " +
                        "to the current pitch. 0 will automatically be appended to this list.", default=[])
    
    parser.add_argument("--min_diff", help="The minimum difference (between language and acoustic) to " +
                        "save a data point. Defaults to 0.", type=float, default=0)
    
    parser.add_argument("--features", help="Use features in the x data points.", action="store_true")
    
    parser.add_argument("--weight", help="This model will output weights directly.", action="store_true")
    
    parser.add_argument("--epochs", help="The number of epochs to train for. Defaults to 100.", type=int, default=100)
    
    parser.add_argument("--no_lstm", help="Do not use the LSTM prior in the data.", action="store_true")
    
    parser.add_argument("--out", help="The directory to save the model to. Defaults to '.' (current directory)",
                        default=".")
    
    args = parser.parse_args()
    
    ac_pitches = args.a_pitches
    ac_pitches.append(0)
    ac_pitches = sorted(list(set(ac_pitches)))
    
    la_pitches = args.l_pitches
    la_pitches.append(0)
    la_pitches = sorted(list(set(la_pitches)))
    
    train_model_full(args.data, history=args.history, ac_pitch_window=ac_pitches,
                     la_pitch_window=la_pitches, min_diff=args.min_diff, features=args.features,
                     out=args.out, is_weight=args.weight, epochs=args.epochs, no_lstm=args.no_lstm)