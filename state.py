import numpy as np

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
            An 88-length vector, representing the state's for the next frame's detections.
        """
        self.hidden_state = hidden_state
        self.prior = prior
        
        self.sample_history = []
        self.prior_history = []
        self.log_prob = 0.0
    
    
    def transition(self, sample, log_prob, hidden_state, prior):
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
            
        Returns
        =======
        state : State
            The state resulting from this transition.
        """
        state = State(hidden_state, prior)
        state.log_prob = self.log_prob + log_prob
        
        state.sample_history = [s for s in self.sample_history]
        state.sample_history.append(sample)
        
        state.prior_history = [p for p in self.prior_history]
        state.prior_history.append(prior)
        return state
        
        
    def get_priors(self):
        """
        Get the priors of this State from each frame.
        
        Returns
        =======
        priors : np.matrix
            An 88 x T matrix containing the priors of this State at each frame.
        """
        priors = np.zeros((88, len(self.prior_history)))
        
        for i, prior in enumerate(self.prior_history):
            priors[:, i] = prior
            
        return priors
    
    
    def get_piano_roll(self):
        """
        Get the piano roll of this State.
        
        Returns
        =======
        priors : np.matrix
            An 88 x T binary matrix containing the pitch detections of this State.
        """
        piano_roll = np.zeros((88, len(self.sample_history)))
        
        for i, sample in enumerate(self.sample_history):
            piano_roll[:, i] = sample
            
        return piano_roll
    