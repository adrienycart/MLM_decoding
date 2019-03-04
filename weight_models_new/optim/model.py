import tensorflow as tf
from tensorflow import keras
import numpy as np
import pickle
import argparse
import datetime
import gzip
import os


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
    """
    with gzip.open(pkl_file, "rb") as file:
        pkl = pickle.load(file)
        
    if history > pkl['history']:
        raise Exception("Desired history length (" + str(history) + ") greater than that saved in data file (" +
                        str(pkl['history']) + ").")
        
    if features and not pkl['features']:
        raise Exception("Wanted to use features, but none are saved in data file.")
    
    ac_pitch_window_np = np.array(ac_pitch_window)
    la_pitch_window_np = np.array(la_pitch_window)
    
    X = []
    Y = pkl['Y'][np.where(pkl['D'] >= min_diff)]
        
    # Doing it in loops because it's easier. If it takes a prohibitavely long time this could be vectorized.
    frame_num_total = -1
    base_index = -88
    for piece_num, acoustic in enumerate(pkl['A']):
        acoustic = np.transpose(acoustic)
        
        for frame_num, frame in enumerate(np.transpose(acoustic)):
            frame_num_total += 1
            
            for beam in range(1 if frame_num == 0 else pkl['beam']):
                base_index += 88
                language = pkl['X'][base_index:base_index + 88, 0:pkl['history']]
                
                for pitch in range(88):
                    x_index = base_index + pitch
                    
                    if pkl['D'][x_index] < min_diff:
                        continue
                    
                    # Usable acoustic pitch window
                    this_pitch_window = ac_pitch_window_np[np.where(np.logical_and(0 <= (pitch + ac_pitch_window_np),
                                                                                  (pitch + ac_pitch_window_np) < 88))]
                    this_pitch_window_index = ac_pitch_window.index(this_pitch_window[0])
                    this_pitch_window_indices = np.array(range(this_pitch_window_index,
                                                      this_pitch_window_index + len(this_pitch_window)))
                    
                    # Usable history length
                    this_history = min(history, frame_num + 1)
                    this_history_index = history - this_history
                    
                    # Acoustic history
                    a = np.zeros((len(ac_pitch_window), history))
                    a[this_pitch_window_indices, this_history_index:] = acoustic[this_pitch_window,
                                                                                 frame_num - this_history + 1:
                                                                                 frame_num + 1]
                    
                    # Usable language pitch window
                    this_pitch_window = la_pitch_window_np[np.where(np.logical_and(0 <= (pitch + la_pitch_window_np),
                                                                                  (pitch + la_pitch_window_np) < 88))]
                    this_pitch_window_index = la_pitch_window.index(this_pitch_window[0])
                    this_pitch_window_indices = np.array(range(this_pitch_window_index,
                                                      this_pitch_window_index + len(this_pitch_window)))
                    
                    
                    # Sample history
                    l = np.zeros((len(la_pitch_window), history))
                    
                    l[this_pitch_window_indices] = language[this_pitch_window, pkl['history'] - history:pkl['history']]
                    
                    f = pkl['X'][x_index, pkl['history']:-2] if features else []
                    
                    X.append(np.hstack((a.reshape(-1), l.reshape(-1), np.squeeze(f), pkl['X'][x_index, -2:])))
          
    X = np.array(X)
    
    if is_weight:
        convert_targets_to_weight(X, Y, history, len(ac_pitch_window))
    
    return X, Y



def convert_targets_to_weight(X, Y, history, ac_pitch_window_size):
    """
    Convert the given targets to train for a prior weight instead of a prior directly.
    
    Y will be edited in place.
    
    Parameters
    ==========
    X : np.ndarray
        An N x (history + num_features + 2) ndarray containing the N data points on which to train.
        
    Y : np.ndarray
        A length-N array, containing the targets for each data point. This will be edited in place.
        
    history : int
        The history length (to find where the acoustic prior is).
        
    ac_pitch_window_size : int
        The acoustic pitch window size (to find where the acoustic prior is).
    """
    la = -1
    
    test = np.zeros((ac_pitch_window_size, history))
    test[int((ac_pitch_window_size - 1) / 2), history-1] = 1
    test = np.squeeze(test.reshape(-1))
    ac = np.where(test == 1)[0][0]

    for i, x in enumerate(X):
        if Y[i] == 0:
            if x[la] < x[ac]:
                Y[i] = 1  #1 trains towards putting weight into the 2nd bin (which is the language weight
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
    return size - history * (ac_pitch_window_size + la_pitch_window_size) - 1



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
    acoustic_in = keras.layers.Input(shape=(ac_pitch_window_size * history,), dtype='float', name='acoustic')
    language_in = keras.layers.Input(shape=(la_pitch_window_size * history,), dtype='float', name='language')
    features_in = keras.layers.Input(shape=(num_features + 1,), dtype='float', name='features')

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



def train_model(model, X, Y, optimizer='adam', epochs=100, checkpoints=[]):
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
        
    optimizer : string OR tf.keras.optimizers.Optimizer
        The optimizer to use when training. Defaults to 'adam', the default-settings Adam optimizer.
        
    epochs : int
        The number of epochs to train for. Defaults to 100.
        
    checkpoints : list(keras.callbacks.ModelCheckpoint)
        A list of checkpoints which will save the model.
    """
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X, Y, epochs=epochs, batch_size=32, callbacks=checkpoints)
    
    
    
    
def train_model_full(data_file, history=5, ac_pitch_window=[-19, -12, 0, 12, 19],
                     la_pitch_window=list(range(-12, 13)), min_diff=0.0, features=True,
                     out="ckpt", is_weight=True, epochs=100):
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
    """
    print("Loading data...")
    X, Y = load_data_from_pkl_file(data_file, min_diff, history, ac_pitch_window, la_pitch_window,
                                   features, is_weight)
    
    ac_pitch_window_size = len(ac_pitch_window)
    la_pitch_window_size = len(la_pitch_window)
    num_features = get_num_features(X.shape[1], history, ac_pitch_window_size, la_pitch_window_size)
    
    acoustic_in = X[:, :len(ac_pitch_window) * history]
    language_in = X[:, len(ac_pitch_window) * history:history * (len(ac_pitch_window) + len(la_pitch_window))]
    features_in = X[:, history * (len(ac_pitch_window) + len(la_pitch_window)):]
    
    print("Loaded " + str(X.shape[0]) + " data points of size " + str(X.shape[1]) + ".")
    print("Training model...")
    
    model = make_model(history, ac_pitch_window_size, la_pitch_window_size, num_features,
                       ac_num_pitch_convs=5, ac_num_history_convs=10, la_num_convs=5)
    
    checkpoint = keras.callbacks.ModelCheckpoint(os.path.join(out, 'model.{epoch:03d}-{loss:.4f}.ckpt'))
    checkpoint_best = keras.callbacks.ModelCheckpoint(os.path.join(out, 'best_loss.ckpt'), monitor='loss',
                                                      save_best_only=True)
    checkpoints = [checkpoint_best, checkpoint]
    
    train_model(model, [acoustic_in, language_in, features_in], Y, epochs=epochs, checkpoints=checkpoints)
    
    model.load_weights(os.path.join(out, 'best_loss.ckpt'))
    model.save('best.h5')
    
    with open(os.path.join(out, 'dict.pkl'), "wb") as file:
        pickle.dump({'model_path' : os.path.join(out, 'best_loss.ckpt'),
                     'history' : history,
                     'ac_pitch_window' : ac_pitch_window,
                     'la_pitch_window' : la_pitch_window,
                     'features' : features,
                     'is_weight' : is_weight},
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
                     out=args.out, is_weight=args.weight, epochs=args.epochs)