import numpy as np
import copy

class State:
    """
    A state in the decoding process, containing a log probability and an LSTM hidden state.
    """

    def __init__(self, P, with_onsets):
        """
        Create a new empty State.
        """
        self.weights = []
        self.combined_prior = []
        self.sample = []
        self.log_prob = 0.0
        self.P = P
        self.with_onsets = with_onsets

        self.num = 0
        self.prev = None
        
        
        
    def update_from_weight_model(self, weights, combined_prior):
        """
        Update this state with data from the weight model.
        
        Parameters
        ==========
        weights : np.array
            An 88-length probabilistic array containing the state's acoustic weights from this frame.

        combined_prior : np.array
            An 88-length probabilistic array containing the combined prior for this frame.
        """
        self.weights = np.repeat(np.array(weights), self.P) if len(weights) == 1 else weights
        self.combined_prior = combined_prior
        
        
        
    def update_from_lstm(self, hidden_state, prior):
        """
        Update this state with data from the LSTM.
        
        Parameters
        ==========
        hidden_state : tf.state
            The hidden state of the LSTM after the previous transition.

        prior : np.array
            An 88-length probabilistic array containing this state's model's prior for the next frame.
        """
        self.hidden_state = hidden_state
        self.prior = np.reshape(prior, -1)
        try:
            self.prev.hidden_state = None
        except:
            pass


    def transition(self, sample, log_prob):
        """
        Get the resulting state from the given transition, without altering this state.

        Parameters
        ==========
        sample : np.array
            An 88-length binary array containing the pitch detections for the following frame.

        log_prob : float
            The log probability of the resulting transition.

        Returns
        =======
        state : State
            The state resulting from this transition.
        """
        state = State(self.P, self.with_onsets)
        state.log_prob = self.log_prob + log_prob

        state.sample = sample
        state.weights = None
        state.combined_prior = None
        state.hidden_state = copy.copy(self.hidden_state)
        state.prior = None
        state.num = self.num + 1
        state.prev = self
        return state


    def get_combined_priors(self):
        """
        Get the combined priors of this State from each frame.

        Returns
        =======
        combined_priors : np.matrix
            A num_pitches x T matrix containing the combined priors of this State at each frame.
        """
        num_pitches = len(self.combined_prior) if self.combined_prior is not None else self.P
        width = self.num if self.combined_prior is not None else self.num-1
        combined_priors = np.zeros((num_pitches, width))

        state = self if self.combined_prior is not None else self.prev
        for i in range(width):
            combined_priors[:, width - 1 - i] = state.combined_prior
            state = state.prev

        return combined_priors


    def get_weights(self):
        """
        Get the weights of this State from each frame.

        Returns
        =======
        weights : np.matrix
            A num_pitches x T matrix containing the weights of this State at each frame.
        """
        num_pitches = len(self.weights) if self.weights is not None else self.P
        width = self.num if self.weights is not None else self.num-1
        weights = np.zeros((num_pitches, width))

        state = self if self.weights is not None else self.prev
        for i in range(width):
            weights[:, width - 1 - i] = state.weights
            state = state.prev

        return weights


    def get_priors(self):
        """
        Get the priors of this State from each frame.

        Returns
        =======
        priors : np.matrix
            A num_pitches x T matrix containing the priors of this State at each frame.
        """
        num_pitches = len(self.prior) if self.prior is not None and len(self.prior) > 0 else self.P
        width = self.num if self.prior is not None else self.num-1
        priors = np.zeros((num_pitches, width + 1))

        state = self if self.prior is not None else self.prev
        for i in range(width + 1):
            priors[:, width - i] = state.prior
            state = state.prev

        return priors


    def get_piano_roll(self, min_length=None, max_length=None, formatter=None):
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
            
        formatter : func(list(int) -> list(int))
            Optionally, a function to convert the samples of this state to another format.

        Returns
        =======
        priors : np.matrix
            A num_pitches x max(min_length, min(T, max_length)) binary matrix containing the pitch
            detections of this State.
        """
        num_pitches = len(self.sample) if self.sample is not None and len(self.sample) > 0 else (self.P // 2 if self.with_onsets else self.P)
        length = min(self.num, max_length) if max_length is not None else self.num
        length = max(min_length, length) if min_length is not None else length
        piano_roll = np.zeros((num_pitches, length))

        state = self
        for i in range(min(length, self.num)):
            piano_roll[:, length - 1 - i] = state.sample
            state = state.prev

        return piano_roll if formatter is None else formatter(piano_roll)

    
def trinary_pr_to_presence_onset(pr):
    """
    Convert from a trinary piano-roll to a presence-onset format one.
    
    Parameters
    ----------
    pr : np.ndarray
        A trinary piano-roll, dimensions (P, T).
        
    Returns
    -------
    binary_pr : np.ndarray
        A binary presence-onset piano-roll, dimensions (2P, T).
    """
    p = len(pr)
    binary_pr = np.zeros((2 * p, pr.shape[1]))
    binary_pr[:p, :] = np.where(pr >= 1, 1, 0)
    binary_pr[p:, :] = np.where(pr == 2, 1, 0)
    return binary_pr