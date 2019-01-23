import state
from mlm_training.model import Model
import numpy as np

class Beam:
    """
    The beam for performing beam search.
    
    This beam stores the current states and has the ability to cut it down
    to a given size, which will save only the most-probable N states.
    """
    
    def __init__(self):
        """
        Create a new beam, initially empty.
        """
        self.beam = []
        
    def __iter__(self):
        """
        Get an iterator for this beam.
        
        Returns
        =======
        iter : iterator
            An iterator for this beam, iterating over the states within it.
        """
        return self.beam.__iter__()
        
    
    def add(self, state):
        """
        Add a State to this beam.
        
        Parameters
        ==========
        state : State
            The state to add to this beam.
        """
        self.beam.append(state)
        
        
        
    def get_top_state(self):
        """
        Get the most probable state from this beam.
        
        Returns
        =======
        The most probable state from the beam.
        """
        best = None
        best_prob = float("-infinity")
        
        for state in self.beam:
            if state.log_prob > best_prob:
                best = state
                best_prob = state.log_prob
                
        return state
        
    
    
    def add_initial_state(self, model, sess):
        """
        Add an empty initial state to the beam.
        
        This is used once before the initial beam search begins.
        
        Parameters
        ==========
        model : Model
            The language model to use for the transduction process
            
        sess : tf.session
            The tensorflow session of the loaded model.
        """
        initial_state = model.get_initial_state(sess, 1)[0]
        prior = np.ones(88) / 2
        
        self.beam.append(state.State(initial_state, prior))
            
        
    def cut_to_size(self, beam_size):
        """
        Removes all but the beam_size most probable states from this beam.
        
        Parameters
        ==========
        beam_size : int
            The maximum number of states to save.
        """
        self.beam = sorted(self.beam, key=lambda s: s.log_prob, reverse=True)[:beam_size]
