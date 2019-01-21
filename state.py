class State:
    """
    A state in the decoding process, containing a log probability and an LSTM hidden state.
    """
    
    def __init__(self, log_prob=0.0, model_state=None, prior=None):
        """
        Create a new State, with the given log_probability and LSTM hidden state.
        
        Parameters
        ==========
        log_prob : float
            The log probability of this state. Defaults to 0.0.
            
        model_state : model state
            The LSTM state associated with this State. Defaults to None, which initializes
            a new LSTM state.
            
        prior : vector
            An 88-length vector, representing the prior for the next frame's detections.
        """
        if model_state is None:
            self.prior, self.model_state = LSTM.get_initial_state()
        else:
            self.model_state = model_state
            self.prior = prior
            
        self.log_prob = log_prob
        
        
        
    def get_next_state(self, sample, log_prob):
        """
        Get the next state we will branch into given a sampled binarized frame.
        
        Parameters
        ==========
        sample : vector
            An 88-length binarized vector, containing pitch detections.
            
        log_prob : float
            The log probability of the given sample.
            
        Returns
        =======
        state : State
            The state reached after the sampled pitch detections.
        """
        model = LSTM.load_state(self.model_state)
        prior, model_state = model.evaluate(sample)
        
        return State(log_prob=self.log_prob + log_prob, model_state=model_state, prior=prior)
