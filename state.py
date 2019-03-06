import numpy as np
import copy

class State:
    """
    A state in the decoding process, containing a log probability and an LSTM hidden state.
    """

    def __init__(self, hidden_state, prior):
        """
        Create a new State, with the given log_probability and LSTM hidden state.

        Parameters
        ==========
        hidden_state : model state
            The LSTM state associated with this State.

        prior : vector
            An 88-length vector, representing the state's prior for the next frame's detections.
        """
        self.hidden_state = hidden_state
        self.prior = prior

        self.weights = []
        self.combined_prior = []
        self.sample = []
        self.log_prob = 0.0

        self.num = 0
        self.prev = None


    def transition(self, sample, log_prob, hidden_state, prior, weights, combined_prior):
        """
        Get the resulting state from the given transition, without altering this state.

        Parameters
        ==========
        sample : np.array
            An 88-length binary array containing the pitch detections for the following frame.

        log_prob : float
            The log probability of the resulting transition.

        hidden_state : tf.state
            The hidden state resulting from the transition.

        prior : np.array
            An 88-length probabilistic array containing this state's model's prior for the next frame.

        weights : np.array
            An 88-length probabilistic array containing the state's acoustic weights from this frame.

        combined_prior : np.array
            An 88-length probabilistic array containing the combined prior for this frame.

        Returns
        =======
        state : State
            The state resulting from this transition.
        """
        state = State(hidden_state, prior)
        state.log_prob = self.log_prob + log_prob

        state.sample = sample
        state.weights = np.repeat(np.array(weights), 88) if len(weights) == 0 else weights
        state.combined_prior = combined_prior
        state.num = self.num + 1
        state.prev = self
        state.prev.hidden_state = None
        return state


    def get_combined_priors(self):
        """
        Get the combined priors of this State from each frame.

        Returns
        =======
        combined_priors : np.matrix
            An 88 x T matrix containing the combined priors of this State at each frame.
        """
        combined_priors = np.zeros((88, self.num))

        state = self
        for i in range(self.num):
            combined_priors[:, self.num - 1 - i] = state.combined_prior
            state = state.prev

        return combined_priors


    def get_weights(self):
        """
        Get the weights of this State from each frame.

        Returns
        =======
        weights : np.matrix
            An 88 x T matrix containing the weights of this State at each frame.
        """
        weights = np.zeros((88, self.num))

        state = self
        for i in range(self.num):
            weights[:, self.num - 1 - i] = state.weights
            state = state.prev

        return weights


    def get_priors(self):
        """
        Get the priors of this State from each frame.

        Returns
        =======
        priors : np.matrix
            An 88 x T matrix containing the priors of this State at each frame.
        """
        priors = np.zeros((88, self.num + 1))

        state = self
        for i in range(self.num + 1):
            priors[:, self.num - i] = state.prior
            state = state.prev

        return priors


    def get_piano_roll(self, min_length=None, max_length=None):
        """
        Get the piano roll of this State.

        Parameters
        ==========
        min_length : int
            The minimum length for a returned piano roll. It will be left-padded with 0s if
            T < min_length. Defaults to None, which does no left padding.

        max_length : int
            The maximum length for a returned piano roll. This will return at most the most recent
            max_length frames.

        Returns
        =======
        priors : np.matrix
            An 88 x max(min_length, min(T, max_length)) binary matrix containing the pitch detections of this State.
        """
        length = min(self.num, max_length) if max_length is not None else self.num
        length = max(min_length, length) if min_length is not None else length
        piano_roll = np.zeros((88, length))

        state = self
        for i in range(min(length, self.num)):
            piano_roll[:, length - 1 - i] = state.sample
            state = state.prev

        return piano_roll
