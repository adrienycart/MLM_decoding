import numpy as np
import itertools
import queue
import argparse
import pretty_midi
import sys
import pickle
import os
import warnings
from tensorflow import keras

import dataMaps
import eval_utils
from beam import Beam
from state import State
from mlm_training.model import Model, make_model_param





def run_lstm_pitchwise_iterative(sess, model, states, uncertainty=0.0):
    """
    Run the LSTM one step, and update the given states in place.

    Parameters
    ==========
    sess : tf.session
        The session for the given model.

    model : Model
        The language model to use for the transduction process.

    states : list(state.State)
        The list containing all of the states we want to update.
    
    uncertainty : float
        The uncertainty of the LSTM prior outputs. The outputs will be scaled
        to a range of size (1 - 2*uncertainty), centered around 0.5. Specifically,
        (0.0, 1.0) -> (0.0+uncertainty, 1.0-uncertainty). Defaults to 0.0.
    """
    samples = np.zeros((len(states), 1, model.n_notes))
    hidden_states = []

    for i, s in enumerate(states):
        hidden_states.append(s.hidden_state)
        samples[i, 0, :] = s.sample

    # Compute next step
    hidden_states, priors = model.run_one_step(hidden_states, samples, sess)

    # Update all states
    for state, hidden_state, prior in zip(states, hidden_states, priors):
        state.update_from_lstm(hidden_state, min(1-uncertainty, max(0+uncertainty, prior)))





def decode_pitchwise(piano_roll, acoustic, model, sess, pitches, beam_size=200, weight=[[0.8], [0.2]],
                     hash_length=10, uncertainty=0.0):
    """
    Transduce the given binary piano roll prior into a binary piano roll by changing the given pitches.

    Parameters
    ==========
    piano_roll : np.ndarray
        A binary piano roll, 88 X T.

    acoustic : np.ndarray
        A probabilistic piano roll, 88 x T, containing values between 0.0 and 1.0
        inclusive. acoustic[p, t] represents the probability of pitch p being present
        at frame t.

    model : Model
        The language model to use for the transduction process.

    sess : tf.session
        The session for the given model.

    pitches : list(int)
        A list of the pitches to transduce.

    beam_size : int
        The beam size for the search. Defaults to 200.

    weight : matrix
        A 2 x 1 matrix, whose first row is the weight for the acoustic model and whose 2nd
        row is the weight for the language model. Defaults to [[0.8], [0.2]].

    hash_length : int
        The history length for the hashed beam. If two states do not differ in the past hash_length
        frames, only the most probable one is saved in the beam. Defaults to 10.

    uncertainty : float
        The uncertainty of the LSTM prior outputs. The outputs will be scaled
        to a range of size (1 - 2*uncertainty), centered around 0.5. Specifically,
        (0.0, 1.0) -> (0.0+uncertainty, 1.0-uncertainty). Defaults to 0.0.

    Returns
    =======
    pr : np.ndarray
        The resulting binary piano roll, 88 X T.
    """
    window = int((model.n_notes - 1) / 2)
    pr_padded = np.vstack((np.zeros((window, piano_roll.shape[1])),
                           piano_roll,
                           np.zeros((window, piano_roll.shape[1]))))

    # One beam per pitch
    beams = []
    for i in range(len(pitches)):
        beam = Beam()
        beam.add_initial_state(model, sess, iterative_pw=True)
        beams.append(beam)

    acoustic = np.transpose(acoustic)

    for frame_num, frame in enumerate(acoustic):
        # Run the LSTM!
        if frame_num != 0:
            run_lstm_pitchwise_iterative(sess, model, [s for beam in beams for s in beam],
                                         uncertainty=uncertainty)

        # Here, the beams contain a list of states, with sample histories, priors, and hidden_states,
        # but needs to be updated with weights and combined_priors when sampling.

        new_beams = []
        for i in range(len(pitches)):
            new_beams.append(Beam())

        # Here we sample from each state in each beam
        for i, (pitch, beam, new_beam) in enumerate(zip(pitches, beams, new_beams)):
            pr_windowed = pr_padded[pitch : pitch + 2 * window + 1, frame_num]

            for state in beam:
                if weight[0][0] == -1:
                    prior = frame[pitch] * state.prior[0]
                    anti_prior = (1-frame[pitch]) * (1-state.prior[0])
                    # print(frame[pitch] , state.prior[0])
                    # print(prior, anti_prior,np.log([prior, anti_prior]))
                else:
                    prior = np.squeeze(weight[0][0] * frame[pitch] + weight[1][0] * state.prior[0])
                    anti_prior = 1-prior
                    # Update state
                    state.update_from_weight_model(weight[0], [prior])


                for log_prob, sample in zip(np.log([prior, anti_prior]), [1, 0]):

                    if window == 0:
                        sample_full = np.array([sample])
                    else:
                        sample_full = np.concatenate((pr_windowed[:window], [sample], pr_windowed[-window:]))

                    # Transition on sample
                    new_beam.add(state.transition(sample_full, log_prob))

            new_beam.cut_to_size(beam_size, min(hash_length, frame_num + 1))
            beams[i] = new_beam

    for i, (pitch, beam) in enumerate(zip(pitches, beams)):
        piano_roll[pitch, :] = beam.get_top_state().get_piano_roll()[window, :]

    return piano_roll





def decode_pitchwise_iterative(acoustic, model, sess, beam_size=200, weight=[[0.8], [0.2]],
                               hash_length=10, out=None, verbose=False, num_iters=5,
                               uncertainty=0.0):
    """
    Transduce the given acoustic piano roll prior into a binary piano roll using iterative pitchwise model.

    Parameters
    ==========
    acoustic : matrix
        A probabilistic piano roll, 88 x T, containing values between 0.0 and 1.0
        inclusive. acoustic[p, t] represents the probability of pitch p being present
        at frame t.

    model : Model
        The language model to use for the transduction process.

    sess : tf.session
        The session for the given model.

    beam_size : int
        The beam size for the search. Defaults to 200.

    weight : matrix
        A 2 x (1 or 88) matrix, whose first row is the weight for the acoustic model and whose 2nd
        row is the weight for the language model, either for each pitch (2x88) or across all pitches
        (2x1). Each column in the matrix should be normalized to sum to 1. Defaults to [[0.8], [0.2]].

    hash_length : int
        The history length for the hashed beam. If two states do not differ in the past hash_length
        frames, only the most probable one is saved in the beam. Defaults to 10.

    out : string
        The directory in which to save the outputs, or None to not save anything. Defaults to None.

    verbose : bool
        Print progress updates. Defaults to False (no printing).

    num_iters : int
        The number of iterations to use, maximum. Defaults to 20.

    uncertainty : float
        The uncertainty of the LSTM prior outputs. The outputs will be scaled
        to a range of size (1 - 2*uncertainty), centered around 0.5. Specifically,
        (0.0, 1.0) -> (0.0+uncertainty, 1.0-uncertainty). Defaults to 0.0.

    Returns
    =======
    prs : list(np.ndarray)
        A list of the resulting binary piano rolls, 88 X T, from each iteration. prs[-1] is the output
        of the system.
    """
    window = int((model.n_notes - 1) / 2)

    if window == 0:
        num_iters = 1

    pr = (acoustic>0.5).astype(int)
    prs = [np.copy(pr)]

    for iteration in range(num_iters):
        if verbose:
            print(f"Iteration {iteration+1}/{num_iters}")

        previous = prs[-1]

        for unlocked_indices in range(window + 1):
            if verbose:
                print(f"Indices {unlocked_indices} mod {window+1}")

            pitches = list(range(unlocked_indices, 88, window+1))

            pr = decode_pitchwise(pr, acoustic, model, sess, pitches, beam_size=beam_size, weight=weight,
                                  hash_length=hash_length, uncertainty=uncertainty)

        diff = int(np.sum(np.abs(previous - pr)))
        print(f"Diff = {diff}")

        if diff == 0:
            break

        prs.append(np.copy(pr))

    return prs






def decode(acoustic, model, sess, branch_factor=50, beam_size=200, weight=[[0.8], [0.2]],
           hash_length=10, out=None, weight_model_dict=None, weight_model=None, verbose=False,
           gt=None):
    """
    Transduce the given acoustic probabilistic piano roll into a binary piano roll.

    Parameters
    ==========
    acoustic : matrix
        A probabilistic piano roll, 88 x T, containing values between 0.0 and 1.0
        inclusive. acoustic[p, t] represents the probability of pitch p being present
        at frame t.

    model : Model
        The language model to use for the transduction process.

    sess : tf.session
        The session for the given model.

    branch_factor : int
        The number of samples to use per frame. Defaults to 50.

    beam_size : int
        The beam size for the search. Defaults to 200.

    weight : matrix
        A 2 x (1 or 88) matrix, whose first row is the weight for the acoustic model and whose 2nd
        row is the weight for the language model, either for each pitch (2x88) or across all pitches
        (2x1). Each column in the matrix should be normalized to sum to 1. Defaults to [[0.8], [0.2]].

    hash_length : int
        The history length for the hashed beam. If two states do not differ in the past hash_length
        frames, only the most probable one is saved in the beam. Defaults to 10.

    out : string
        The directory in which to save the outputs, or None to not save anything. Defaults to None.

    weight_model_dict : dict
        A dictionary containing information about the weight model to use, if any. Defaults to None,
        which uses the static weight of the weight parameter.

    weight_model : sklearn.model or tf.keras.Model
        The model to be used as a weight_model, or None to use static weighting.

    verbose : bool
        Print progress in number of frames. Defaults to False (no printing).

    gt : matrix
        The ground truth piano roll, 88 x T. If given, this will be used to always use the optimum
        weight for each frame. Defaults to None.


    Returns
    =======
    piano_roll : np.ndarray
        An 88 x T binary piano roll, where a 1 represents the presence of a pitch
        at a given frame.

    priors : np.ndarray
        An 88 x T matrix, giving the prior assigned to each pitch detection by the
        most probable language model state.

    weights : np.ndarray
        An 88 X T matrix, giving the acoustic weights for each pitch at each frame.
    """
    if gt is not None:
        weight_model = True
        is_weight = True

    if (not weight_model) and weight[0][0] == 1.0:
        return (acoustic>0.5).astype(int), np.zeros(acoustic.shape), np.ones(acoustic.shape), acoustic

    weights_all = None
    priors_all = None

    beam = Beam()
    beam.add_initial_state(model, sess)

    acoustic = np.transpose(acoustic)

    for frame_num, frame in enumerate(acoustic):
        if verbose and frame_num % 20 == 0:
            print(str(frame_num) + " / " + str(acoustic.shape[0]))

        # Run the LSTM!
        if frame_num != 0:
            run_lstm(sess, model, beam)

        # Here, beam contains a list of states, with sample histories, priors, and LSTM hidden_states,
        # but needs to be updated with weights and combined_priors when sampling.

        # Here, we are calculating dynamic weights or priors if we are using gt or a weight_model
        if weight_model:
            weights_all, priors_all = run_weight_model(gt, weight_model, weight_model_dict, beam,
                                                       acoustic, frame_num)

        new_beam = Beam()

        # Here we sample from each state in the beam
        for i, state in enumerate(beam):
            weight_this = weights_all[:, i * 88 : (i + 1) * 88] if weights_all is not None else weight

            if priors_all is not None:
                prior = np.squeeze(priors_all[i * 88 : (i + 1) * 88])
            else:
                prior = np.squeeze(weight_this[0] * frame + weight_this[1] * state.prior)

            # Update state
            state.update_from_weight_model(weight_this[0], prior)

            for log_prob, sample in itertools.islice(enumerate_samples(prior), branch_factor):

                # Binarize the sample (return from enumerate_samples is an array of indexes)
                binary_sample = np.zeros(88)
                binary_sample[sample] = 1

                # Transition on sample
                new_beam.add(state.transition(binary_sample, log_prob))

        new_beam.cut_to_size(beam_size, min(hash_length, frame_num + 1))
        beam = new_beam

        if out:
            output = [(s.get_piano_roll(), s.get_priors(), s.get_weights(), s.get_combined_priors()) for s in beam]
            with open(os.path.join(out, 'data_' + str(frame_num) + '.pkl'), 'wb') as file:
                pickle.dump(output, file)

    top_state = beam.get_top_state()
    return top_state.get_piano_roll(), top_state.get_priors(), top_state.get_weights(), top_state.get_combined_priors()




def run_weight_model(gt, weight_model, weight_model_dict, beam, acoustic, frame_num):
    """
    Run the weight_model and return its results.

    Parameters
    ==========
    gt : matrix
        The ground truth piano roll, 88 x T. If given, this will be used to always use the optimum
        weight for each frame.

    weight_model : sklearn.model or tf.keras.Model
        The model to be used as a weight_model.

    weight_model_dict : dict
        A dictionary containing information about the weight model to use.

    beam : beam.Beam
        The beam containing all of the states to get data from.

    acoustic : np.ndarray
        A Tx88 array, the acoustic prior for each pitch at each frame.

    frame_num : int
        The frame number we are currently on.

    Returns
    =======
    weights_all : np.ndarray
        A 2 x (88*beam) array, containing the acoustic (index 0) and language (index 1) weights
        for each sample and pitch.

    priors_all : np.array
        An (88*beam)-length array, containing the prior for each sample and pitch.
    """
    if gt:
        weights_all = np.transpose(np.vstack([get_best_weights(state.prior, frame, gt[:, frame_num]) for state in beam]))
        priors_all = None
        return weights_all, priors_all

    # Load the weight_model properties
    if 'model' in weight_model_dict:
        sklearn = True
        history = weight_model_dict['history']
        features = weight_model_dict['features'] if 'features' in weight_model_dict else False
        is_weight = weight_model_dict['weight'] if 'weight' in weight_model_dict else True
        history_context = weight_model_dict['history_context'] if 'history_context' in weight_model_dict else 0
        prior_context = weight_model_dict['prior_context'] if 'prior_context' in weight_model_dict else 0
        use_lstm = weight_model_dict['use_lstm'] if 'use_lstm' in weight_model_dict else True
    else:
        sklearn = False
        history = weight_model_dict['history']
        ac_pitch_window = weight_model_dict['ac_pitch_window']
        la_pitch_window = weight_model_dict['la_pitch_window']
        features = weight_model_dict['features']
        is_weight = weight_model_dict['is_weight']
        no_lstm = weight_model_dict['no_lstm'] if 'no_lstm' in weight_model_dict else False

    if sklearn:
        X = np.vstack([create_weight_x_sk(state, acoustic, frame_num, history, features=features,
                                          history_context=history_context,
                                          prior_context=prior_context) for state in beam])
        # Remove LSTM sample
        if not use_lstm:
            X = X[:, :-1]

        # 2 x len(X) matrix
        weights_all = np.transpose(weight_model.predict_proba(X)) if is_weight else None

        # len(X) array
        priors_all = np.squeeze(weight_model.predict_proba(X)[:, 1]) if not is_weight else None

    else: # tensorflow
        X = np.vstack([create_weight_x_tf(state, acoustic, frame_num, history, ac_pitch_window,
                                          la_pitch_window, features) for state in beam])
        # Remove LSTM sample
        if no_lstm:
            X = X[:, :-1]

        # Split X data into its parts
        acoustic_in = X[:, :len(ac_pitch_window) * history].reshape(-1, len(ac_pitch_window), history)
        language_in = X[:, len(ac_pitch_window) * history:
                        history * (len(ac_pitch_window) + len(la_pitch_window))
                       ].reshape(-1, len(la_pitch_window), history)
        features_in = X[:, history * (len(ac_pitch_window) + len(la_pitch_window)):]
        X_split = [acoustic_in, language_in, features_in]

        result = np.squeeze(weight_model.predict(X_split))

        # 2 x len(X) matrix
        weights_all = None
        if is_weight:
            weights_all = np.zeros((2, len(X)))
            weights_all[1, :] = result
            weights_all[0, :] = 1 - result

        # len(X) array
        priors_all = np.squeeze(result) if not is_weight else None

    return weights_all, priors_all




def run_lstm(sess, model, beam):
    """
    Run the LSTM one step, and update the states in the beam in place.

    Parameters
    ==========
    sess : tf.session
        The session for the given model.

    model : Model
        The language model to use for the transduction process.

    beam : beam.Beam
        The beam containing all of the states we want to update.
    """
    if model.pitchwise:
        window = int((model.n_notes - 1) / 2)

        hidden_states_in = []
        pw_samples = np.zeros((len(beam) * 88, 1, model.n_notes))

        for i, s in enumerate(beam):
            hidden_states_in.extend(s.hidden_state)

            sample_pad = np.pad(s.sample, (window, window), 'constant')
            for j in range(window, window + 88):
                pw_samples[i * 88 + j - window, 0, :] = sample_pad[j - window : j + window + 1]

        # Compute next step
        hidden_states_out, pw_priors = model.run_one_step(hidden_states_in, pw_samples, sess)

        # Update all states
        for i, s in enumerate(beam):
            hidden_state = hidden_states_out[88 * i : 88 * (i+1)]
            prior = np.squeeze(pw_priors[88 * i : 88 * (i+1)])

            s.update_from_lstm(hidden_state, prior)

    else: # Not pitchwise
        hidden_states = []
        np_samples = np.zeros((len(beam), 1, 88))

        # Get states
        for i, s in enumerate(beam):
            hidden_states.append(s.hidden_state)
            np_samples[i, 0, :] = s.sample

        # Run LSTM
        hidden_states, priors = model.run_one_step(hidden_states, np_samples, sess)

        # Update states
        for i, s in enumerate(beam):
            s.update_from_lstm(hidden_states[i], priors[i])




def get_best_weights(language, acoustic, gt, width=0.25):
    """
    Get the best weights for the given priors and ground truth.

    Parameters
    ==========
    language : np.array
        The language model priors, of length N.

    acoustic : np.array
        The acoustic model priors, of length N.

    gt : np.array
        The ground truth outputs, a binrazed array of length N.

    Returns
    =======
    weights : np.ndarray
        An N x 2 array with the best weights for each sample, on the range [0.0, 1.0].
        The 1st column is the acoustic weight, and the 2nd column is the language weight.
    """
    language = np.squeeze(language)

    weights = np.zeros((len(language), 2))

    language_diffs = np.abs(language - gt)
    acoustic_diffs = np.abs(acoustic - gt)

    weights[:, 0] = np.where(acoustic_diffs < language_diffs,
                             np.random.uniform(low=1.0-width, high=1.0, size=(len(weights),)),
                             np.random.uniform(low=0.0, high=0.0+width, size=(len(weights),)))
    weights[:, 1] = 1 - weights[:, 0]

    return weights




def create_weight_x_tf(state, acoustic, frame_num, history, ac_pitch_window, la_pitch_window, features):
    """
    Get the x input for the tf.keras dynamic weighting model.

    Parameters
    ==========
    state : State
        The state to examine for its piano roll and prior.

    acoustic : np.ndarray
        The acoustic prior for the entire piece.

    frame_num : int
        The current frame number.

    history : int
        The number of frames to look back in time for both acoustic and language history.

    ac_pitch_window : list(int)
        The pitches around each data point to use, for its acoustic history.

    la_pitch_window : list(int)
        The pitches around each data point to use, for its sample history.

    features : boolean
        True to calculate features. False otherwise.

    Return
    ======
    x : np.ndarray
        The x data points for the given input for the dynamic weighting model.
    """
    frame = acoustic[frame_num, :]
    pr = state.get_piano_roll(min_length=history, max_length=history)

    x = np.array(get_data_tf(history, ac_pitch_window, la_pitch_window, np.transpose(acoustic),
                             pr, frame_num, list(range(88))))

    if features:
        x = np.hstack((x,
                       get_features(acoustic, frame_num, state.get_priors()),
                       np.reshape(frame, (88, -1)),
                       np.reshape(state.prior, (88, -1))))
    else:
        x = np.hstack((x,
                       np.reshape(frame, (88, -1)),
                       np.reshape(state.prior, (88, -1))))

    return x



def get_data_tf(history, ac_pitch_window, la_pitch_window, acoustic, pr, frame_num, pitches):
    """
    Get the history-based features for some tensor flow data points.

    Parameters
    ==========
    history : int
        The number of frames to look back in time for both acoustic and language history.

    ac_pitch_window : list(int)
        The pitches around each data point to use, for its acoustic history.

    la_pitch_window : list(int)
        The pitches around each data point to use, for its sample history.

    acoustic : np.ndarray
        An 88xN array, representing the acoustic prior for this entire piece.

    pr : np.ndarray
        An 88x(frame_num-1) array of the binary samples of this piece so far.

    frame_num : int
        The frame number we are on.

    pitches : list(int)
        The pitches we want data from.

    Returns
    =======
    data : list
        A list of of the history-based features for the desired data points.
    """
    frame = acoustic[:, frame_num]

    ac_pitch_window_np = np.array(ac_pitch_window)
    la_pitch_window_np = np.array(la_pitch_window)

    x = []

    for pitch in pitches:
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
        l[this_pitch_window_indices] = pr[this_pitch_window, -history:]

        x.append(np.hstack((a.reshape(-1), l.reshape(-1))))

    return x




def create_weight_x_sk(state, acoustic, frame_num, history, pitches=range(88), features=False,
                    history_context=0, prior_context=0):
    """
    Get the x input for the sk-learn dynamic weighting model.

    Parameters
    ==========
    state : State
        The state to examine for its piano roll and prior.

    acoustic : np.ndarray
        The acoustic prior for the entire piece, as time X pitch.

    frame_num : int
        The current frame number.

    history : int
        How many frames to save in the x data point.

    pitches : list
        The pitches we want data points for. Defaults to [0:88] (all pitches).

    features : boolean
        True to calculate features. False otherwise.

    history_context : int
        The pitch window to include around the history of samples, unrolled, and 0 padded.
        Defaults to 0.

    prior_context : int
        The window of priors to include around the current priors. Defaults to 0.

    Returns
    =======
    x : np.ndarray
        The x data points for the given input for the dynamic weighting model.
    """
    frame = acoustic[frame_num, :]
    pr = state.get_piano_roll(min_length=history, max_length=history)

    if features:
        x = np.hstack((pr,
                       get_features(acoustic, frame_num, state.get_priors()),
                       np.reshape(frame, (88, -1)),
                       np.reshape(state.prior, (88, -1))))

    else:
        x = np.hstack((pr,
                       np.reshape(frame, (88, -1)),
                       np.reshape(state.prior, (88, -1))))

    # Add prior and history contexts
    x_new = pad_x(x, frame, state.prior, pr, history, history_context, prior_context)

    return x_new[pitches]



def pad_x(x, acoustic, language, pr, history, history_context, prior_context):
    """
    Add a pitch and acoustic prior window around its given history to an sk-learn x data point.

    Parameters
    ==========
    x : np.ndarray
        The original data points, num_data_points X num_features.

    acoustic : np.ndarray
        The acoustic prior for the entire piece, as time X pitch.

    language : np.ndarray
        The language model priors of the past steps, as an 88 X N array.

    pr : np.ndarray
        The binary piano roll of the past steps, as an 88 X N array.

    history_context : int
        The window radius around the samples to save.

    prior_context : int
        The window radius around the acoustic priors to save.

    Returns
    =======
    x_new : np.ndarray
        The padded data points.
    """
    x_new = np.zeros((x.shape[0], x.shape[1] + prior_context * 4 + 2 * history_context * history))
    x_new[:, :x.shape[1]] = x

    extra_start = x.shape[1]

    if prior_context != 0:
        acoustic_padded = np.zeros(88 + prior_context * 2)
        acoustic_padded[prior_context:-prior_context] = acoustic

        language_padded = np.zeros(88 + prior_context * 2)
        language_padded[prior_context:-prior_context] = language

        for i in range(prior_context):
            x_new[:, extra_start + i] = acoustic_padded[i:-2 * prior_context + i]
            x_new[:, extra_start + prior_context + i] = language_padded[i:-2 * prior_context + i]
            if i == 0:
                x_new[:, extra_start + 2 * prior_context + i] = acoustic_padded[2 * prior_context - i:]
                x_new[:, extra_start + 3 * prior_context + i] = language_padded[2 * prior_context - i:]
            else:
                x_new[:, extra_start + 2 * prior_context + i] = acoustic_padded[2 * prior_context - i:-i]
                x_new[:, extra_start + 3 * prior_context + i] = language_padded[2 * prior_context - i:-i]

    extra_start += 4 * prior_context

    if history_context != 0 and history != 0:
        pr_padded = np.zeros((88 + history_context * 2, history))
        pr_padded[history_context:-history_context, :] = pr

        for i in range(history_context):
            x_new[:, extra_start + i * history :
                  extra_start + (i + 1) * history] = pr_padded[i:-2 * history_context + i, :]
            if i == 0:
                x_new[:, extra_start + history_context * history + i * history :
                      extra_start + history_context * history + (i + 1) * history] = pr_padded[2 * history_context - i:, :]
            else:
                x_new[:, extra_start + history_context * history + i * history :
                      extra_start + history_context * history + (i + 1) * history] = pr_padded[2 * history_context - i:-i, :]

    return x_new



def get_features(acoustic, frame_num, priors):
    """
    Get a features array from the given acoustic and language model priors.

    Parameters
    ==========
    acoustic : np.ndarray
        The acoustic prior for the entire piece.

    frame_num : int
        The current frame number.

    language : np.ndarray
        The language priors from the entire piece.

    Returns
    =======
    features : np.ndarray
        A 88 x (num_features) array.
    """
    def uncertainty(array):
        """
        Get the average squared error of each number in an array's distance from certainty (1 or 0).

        Parameters
        ==========
        array : np.array
            Any data array.

        Returns
        =======
        uncertainty : float
            The average squared error of each element in the array's difference from certainty (1 or 0).
        """
        normed = np.minimum(array, np.abs(1 - array))
        return np.mean(normed * normed)

    def entropy(array):
        """
        Get the entropy of the given array.

        Parameters
        ==========
        array : np.array
            An data array.

        Returns
        =======
        entropy : float
            The entropy of the given array. A measure of its flatness.
        """
        return np.sum(np.where(array == 0, 0, -array * np.log2(array))) / np.log2(len(array))

    num_features = 9
    frame = acoustic[frame_num, :]
    language = np.squeeze(priors[:, -1])
    features = np.zeros((88, num_features))

    features[:, 0] = uncertainty(acoustic)
    features[:, 1] = uncertainty(language)
    features[:, 2] = entropy(acoustic)
    features[:, 3] = entropy(language)
    features[:, 4] = np.mean(acoustic)
    features[:, 5] = np.mean(language)

    # Flux
    if frame_num != 0:
        features[:, 6] = frame - acoustic[frame_num-1, :]
        features[:, 7] = language - priors[:, -2]
    else:
        features[:, 6] = frame
        features[:, 7] = language

    # Absolute pitch (0, 1) range
    features[:, 8] = np.arange(88) / 87

    return features




def get_log_prob(sample, acoustic, language, weight, p=None):
    """
    Get the log probability of a set of samples given the priors and weights.

    Parameters
    ==========
    sample : np.ndarray
        An N x 88 matrix representing N possible samples.

    acoustic : np.array
        An 88-length array, containing the probability of each pitch being present,
        according to the acoustic model.

    language : np.ndarray
        An N x 88 matrix containing the probability of each pitch being present,
        according to each sampled state.

    weight : np.ndarray
        An N x 2 x (1 or 88) size tensor, whose first index corresponds to each of the N samples,
        second index corresponds to the prior (index 0 for acoustic prior, index 1 for language prior),
        and whose third dimension is either length 1 (when each pitch has the same prior) or length
        88 (when each pitch has a different prior).

    p : np.array
        A weighted probability prior for each pitch. This overrides all other arguments
        to be used as p if it is given. Defaults to None.

    Returns
    =======
    log_prob : np.array
        The log probability of each given sample, as a weighted sum of p(. | acoustic) and p(. | language),
        in an N x 1 array.

    combined_priors : np.ndarray
        The combined (NOT log) prior of each given sample, in an N x 88 nd-array.
    """
    if p is None:
        weight_acoustic = np.squeeze(weight[:, 0, :]) # N or N x 88
        weight_language = np.squeeze(weight[:, 1, :]) # N or N x 88

        if np.ndim(weight_acoustic) == 1:
            p = np.outer(weight_acoustic, acoustic) + language * np.reshape(weight_language, (-1, 1))
        else:
            p = weight_acoustic * acoustic + weight_language * language # N x 88
    else:
        p = np.squeeze(p)

    not_p = 1 - p

    return np.sum(np.log(np.where(sample == 1, p, not_p)), axis=1), p




def enumerate_samples(p):
    """
    Enumerate the binarised piano-roll samples of a frame, ordered by probability.

    Based on Algorithm 2 from:
    Boulanger-Lewandowski, Bengio, Vincent - 2013 - High-dimensional Sequence Transduction

    Parameters
    ==========
    p : np.array
        A weighted probability prior for each pitch.

    Yields
    ======
    log_prob : float
        The log probability of the given sample.

    samples : list(int)
        A list of the indices where there should be 1's in tthe samples.
    """
    not_p = 1 - p

    # Base case: most likely chosen greedily
    v_0 = np.where(p > not_p)[0]
    l_0 = np.sum(np.log(np.maximum(p, not_p)))
    yield l_0, v_0

    v_0 = set(v_0)

    # Sort likelihoods by likelihood penalty for flipping
    L = np.abs(np.log(p / not_p))
    R = L.argsort()
    L_sorted = L[R]

    # Num solves crash for duplicate likelihoods
    num = 1
    q = queue.PriorityQueue()
    q.put((L_sorted[0], 0, [0]))

    while not q.empty():
        l, _, v = q.get()
        yield l_0 - l, list(v_0.symmetric_difference(R[v]))

        i = np.max(v)
        if i + 1 < len(L_sorted):
            v.append(i + 1)
            q.put((l + L_sorted[i + 1], num, v))
            v = v.copy()

            # XOR between v and [i]
            try:
                v.remove(i)
            except:
                v.append(i)

            q.put((l + L_sorted[i + 1] - L_sorted[i], num+1, v))
            num += 2






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("MIDI", help="The MIDI file to load. This should be in the same location as the " +
                        "corresponding acoustic csv, and have the same name (except the extension).")

    parser.add_argument("-m", "--model", help="The location of the trained language model.", required=True)
    parser.add_argument("--hidden", help="The number of hidden layers in the language model. Defaults to 256",
                        type=int, default=256)

    parser.add_argument("--step", type=str, choices=["time", "quant", "event"], help="Change the step type " +
                        "for frame timing. Either time (default), quant (for 16th notes), or event (for onsets).",
                        default="time")

    parser.add_argument("-b", "--beam", help="The beam size. Defaults to 100.", type=int, default=100)
    parser.add_argument("-k", "--branch", help="The branching factor. Defaults to 20.", type=int, default=20)

    parser.add_argument("-w", "--weight", help="The weight for the acoustic model (between 0 and 1). " +
                        "Defaults to 0.8", type=float, default=0.8)
    parser.add_argument("-wm", "--weight_model", help="Load the given sklearn model using pickle, to dynamically " +
                        "set weights. Defaults to None, which uses the static weight from -w instead.",
                        default=None)

    parser.add_argument("--max_len",type=str,help="test on the first max_len seconds of each text file. " +
                        "Anything other than a number will evaluate on whole files. Default is 30s.",
                        default=30)

    parser.add_argument("-o", "--output", help="The directory to save outputs to. Defaults to None (don't save).",
                        default=None)
    parser.add_argument("--hash", help="The hash length to use. Defaults to 10.",
                        type=int, default=10)
    parser.add_argument("-v", "--verbose", help="Print frame status updates.", action="store_true")
    parser.add_argument("--gpu", help="The gpu to use. Defaults to 0.", default="0")

    parser.add_argument("--gt", help="Use the gt to use the best possible weight_model results.", action="store_true")

    parser.add_argument("--pitchwise", type=int, help="Use the pitchwise language model. Give the window size.")

    parser.add_argument("--it", help="Use iterative pitchwise processing with this number of iterations. " +
                        "Defaults to 0, which doesn't use iterative processing.", type=int, default=0)

    parser.add_argument("--uncertainty", help="Add some uncertainty to the LSTM decoding outputs, when " +
                        "used with --it. The outputs will be scaled to a range of size " +
                        "(1 - 2*uncertainty), centered around 0.5. Specifically, " +
                        "(0.0, 1.0) -> (0.0+uncertainty, 1.0-uncertainty). Defaults to 0.0.",
                        type=float, default=0)

    args = parser.parse_args()

    if not (0 <= args.weight <= 1):
        print("Weight must be between 0 and 1.", file=sys.stderr)
        sys.exit(1)

    warnings.filterwarnings("ignore", message="tick should be an int.")

    try:
        max_len = float(args.max_len)
        section = [0, max_len]
    except:
        max_len = None
        section = None

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Load data
    data = dataMaps.DataMaps()
    data.make_from_file(args.MIDI, args.step, section=section, acoustic_model='kelz')

    # Load model
    n_hidden = args.hidden

    # Load model
    model_param = make_model_param()
    model_param['n_hidden'] = n_hidden
    model_param['n_steps'] = 1 # To generate 1 step at a time
    if args.pitchwise is None:
        model_param['pitchwise'] = False
    else:
        model_param['pitchwise'] = True
        model_param['n_notes'] = 2 * args.pitchwise + 1

    # Build model object
    model = Model(model_param)
    sess,_ = model.load(args.model, model_path=args.model)

    # Load weight model
    weight_model_dict = None
    weight_model = None
    if args.weight_model:
        with open(args.weight_model, "rb") as file:
            weight_model_dict = pickle.load(file)
        if 'model' in weight_model_dict:
            weight_model = weight_model_dict['model']
        else:
            weight_model = keras.models.load_model(weight_model_dict['model_path'])

    if args.output is not None:
        os.makedirs(args.output, exist_ok=True)

    # Decode
    if args.it > 0:
        prs = decode_pitchwise_iterative(data.input, model, sess, beam_size=args.beam,
                                         weight=[[args.weight], [1 - args.weight]],
                                         hash_length=args.hash, verbose=args.verbose, num_iters=args.it,
                                         uncertainty=args.uncertainty)

        pr = prs[-1]

        # Evaluate
        if args.output is not None:
            np.save(os.path.join(args.output, "prs"), prs)
        if args.step in ['quant','event']:
            pr = dataMaps.convert_note_to_time(pr, data.corresp, data.input_fs, max_len=max_len)

    else:
        pr, priors, weights, combined_priors = decode(data.input, model, sess, branch_factor=args.branch,
                        beam_size=args.beam, weight=[[args.weight], [1 - args.weight]],
                        out=None, hash_length=args.hash, weight_model_dict=weight_model_dict,
                        verbose=args.verbose, gt=data.target if args.gt else None, weight_model=weight_model)

        # Evaluate
        if args.output is not None:
            np.save(os.path.join(args.output, "pr"), pr)
            np.save(os.path.join(args.output, "priors"), priors)
            np.save(os.path.join(args.output, "weights"), weights)
            np.save(os.path.join(args.output, "combined_priors"), combined_priors)
        if args.step in ['quant','event']:
            pr = dataMaps.convert_note_to_time(pr, data.corresp, max_len=max_len)

    data = dataMaps.DataMaps()
    data.make_from_file(args.MIDI, "time", section=section)
    target = data.target

    P_f, R_f, F_f = eval_utils.compute_eval_metrics_frame(pr, target)
    P_n, R_n, F_n = eval_utils.compute_eval_metrics_note(pr, target, min_dur=0.05)

    print(f"Frame P,R,F: {P_f:.3f},{R_f:.3f},{F_f:.3f}, Note P,R,F: {P_n:.3f},{R_n:.3f},{F_n:.3f}")
