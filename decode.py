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
import sampling
from beam import Beam
from state import State
from mlm_training.model import Model, make_model_param





def decode(acoustic, model, sess, branch_factor=50, beam_size=200, weight=[[0.8], [0.2]],
           hash_length=10, out=None, weight_model_dict=None, weight_model=None, verbose=False,
           gt=None):
    """
    Transduce the given acoustic probabilistic piano roll into a binary piano roll.

    Parameters
    ==========
    acoustic : matrix
        A probabilistic piano roll, P x T, containing values between 0.0 and 1.0
        inclusive. acoustic[p, t] represents the probability of pitch p being present
        at frame t. Or, if model.with_onsets is true, the bottom-half of this is onsets.

    model : Model
        The language model to use for the transduction process.

    sess : tf.session
        The session for the given model.

    branch_factor : int
        The number of samples to use per frame. Defaults to 50.

    beam_size : int
        The beam size for the search. Defaults to 200.

    weight : matrix
        A 2 x (1 or P) matrix, whose first row is the weight for the acoustic model and whose 2nd
        row is the weight for the language model, either for each pitch (2xP) or across all pitches
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
        The ground truth piano roll, P x T. If given, this will be used to always use the optimum
        weight for each frame. Defaults to None.


    Returns
    =======
    piano_roll : np.ndarray
        An P x T binary piano roll, where a 1 represents the presence of a pitch
        at a given frame.

    priors : np.ndarray
        An P x T matrix, giving the prior assigned to each pitch detection by the
        most probable language model state.

    weights : np.ndarray
        An P X T matrix, giving the acoustic weights for each pitch at each frame.
    """
    P = len(acoustic)
    
    if gt is not None:
        weight_model = True
        is_weight = True

    if (not weight_model) and weight[0][0] == 1.0:
        return (acoustic>0.5).astype(int), np.zeros(acoustic.shape), np.ones(acoustic.shape), acoustic

    weights_all = None
    priors_all = None

    beam = Beam()
    beam.add_initial_state(model, sess, P)

    acoustic = np.transpose(acoustic)
    
    lstm_transform = None
    if model.with_onsets:
        lstm_transform = three_hot_output_to_presence_onset

    for frame_num, frame in enumerate(acoustic):
        if verbose and frame_num % 20 == 0:
            print(str(frame_num) + " / " + str(acoustic.shape[0]))

        # Run the LSTM!
        if frame_num != 0:
            run_lstm(sess, model, beam, P, transform=lstm_transform)

        # Here, beam contains a list of states, with sample histories, priors, and LSTM hidden_states,
        # but needs to be updated with weights and combined_priors when sampling.

        # Here, we are calculating dynamic weights or priors if we are using gt or a weight_model
        if weight_model:
            weights_all, priors_all = run_weight_model(gt, weight_model, weight_model_dict, beam,
                                                       acoustic, frame_num)

        new_beam = Beam()

        # Here we sample from each state in the beam
        for i, state in enumerate(beam):
            weight_this = weights_all[:, i * P : (i + 1) * P] if weights_all is not None else weight

            if priors_all is not None:
                prior = np.squeeze(priors_all[i * P : (i + 1) * P])
            else:
                prior = np.squeeze(weight_this[0] * frame + weight_this[1] * state.prior)

            # Update state
            state.update_from_weight_model(weight_this[0], prior)

            for log_prob, sample in itertools.islice(sampling.enumerate_samples(prior), branch_factor):

                # Format the sample (return from enumerate_samples is an array of indexes)
                if model.with_onsets:
                    sample = sampling.trinarize_with_onsets(sample, P)
                else:
                    sample = sampling.binarize(sample, P)

                # Transition on sample
                new_beam.add(state.transition(sample, log_prob))

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
        The ground truth piano roll, P x T. If given, this will be used to always use the optimum
        weight for each frame.

    weight_model : sklearn.model or tf.keras.Model
        The model to be used as a weight_model.

    weight_model_dict : dict
        A dictionary containing information about the weight model to use.

    beam : beam.Beam
        The beam containing all of the states to get data from.

    acoustic : np.ndarray
        A TxP array, the acoustic prior for each pitch at each frame.

    frame_num : int
        The frame number we are currently on.

    Returns
    =======
    weights_all : np.ndarray
        A 2 x (P*beam) array, containing the acoustic (index 0) and language (index 1) weights
        for each sample and pitch.

    priors_all : np.array
        An (P*beam)-length array, containing the prior for each sample and pitch.
    """
    if gt:
        weights_all = np.transpose(np.vstack([get_best_weights(state.prior, frame, gt[:, frame_num]) for state in beam]))
        priors_all = None
        return weights_all, priors_all

    # Load the weight_model properties
    history = weight_model_dict['history']
    features = weight_model_dict['features'] if 'features' in weight_model_dict else False
    is_weight = weight_model_dict['weight'] if 'weight' in weight_model_dict else True
    history_context = weight_model_dict['history_context'] if 'history_context' in weight_model_dict else 0
    prior_context = weight_model_dict['prior_context'] if 'prior_context' in weight_model_dict else 0
    use_lstm = weight_model_dict['use_lstm'] if 'use_lstm' in weight_model_dict else True
    no_mlm = weight_model_dict['no_mlm'] if 'no_mlm' in weight_model_dict else False

    X = np.vstack([create_weight_x_sk(state, acoustic, frame_num, history, features=features,
                                      history_context=history_context,
                                      prior_context=prior_context,
                                      no_mlm=no_mlm) for state in beam])
    # Remove LSTM sample
    if not use_lstm:
        X = X[:, :-1]

    # 2 x len(X) matrix
    weights_all = np.transpose(weight_model.predict_proba(X)) if is_weight else None

    # len(X) array
    priors_all = np.squeeze(weight_model.predict_proba(X)[:, 1]) if not is_weight else None

    return weights_all, priors_all




def run_lstm(sess, model, beam, P, transform=None):
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
    
    transform : function(list(float) -> list(float))
    """
    hidden_states = []
    np_samples = np.zeros((len(beam), 1, len(beam.get_top_state().sample)))

    # Get states
    for i, s in enumerate(beam):
        hidden_states.append(s.hidden_state)
        np_samples[i, 0, :] = s.sample

    # Run LSTM
    hidden_states, priors = model.run_one_step(hidden_states, np_samples, sess)
    
    # Transfor the LSTM prior into a different format if necessary
    if transform is not None:
        priors = transform(priors)

    # Update states
    for i, s in enumerate(beam):
        s.update_from_lstm(hidden_states[i], priors[i])


def three_hot_output_to_presence_onset(priors):
    """
    Convert from a three-hot LSTM output to the presence-onset format.
    
    Parameters
    ----------
    priors : np.ndarray
        A dim (?, 1, 88, 3) array of LSTM priors.
        
    Returns
    -------
    priors : np.ndarray
        A dim (?, 88*2) array of presence-onset format.
    """
    return priors[:, :, :, 1:].reshape(len(priors), -1)


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




def create_weight_x_sk(state, acoustic, frame_num, history, pitches=range(88), features=False,
                    history_context=0, prior_context=0, no_mlm=False):
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

    no_mlm : boolean
        Whether to suppress MLM-based inputs. Defaults to False.

    Returns
    =======
    x : np.ndarray
        The x data points for the given input for the dynamic weighting model.
    """
    frame = acoustic[frame_num, :]
    pr = state.get_piano_roll(min_length=history, max_length=history)

    if features:
        x = np.hstack((pr,
                       get_features(acoustic, frame_num, state.get_priors(), no_mlm=no_mlm),
                       np.reshape(frame, (88, -1)),
                       np.zeros((88, 1)) if no_mlm else np.reshape(state.prior, (88, -1))))

    else:
        x = np.hstack((pr,
                       np.reshape(frame, (88, -1)),
                       np.zeros((88, 1)) if no_mlm else np.reshape(state.prior, (88, -1))))

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



def get_features(acoustic, frame_num, priors, no_mlm=False):
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

    no_mlm : boolean
        Whether to suppress MLM-based inputs. Defaults to False.

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

    if no_mlm:
        features[:, [1,3,5,7]] = 0

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
            