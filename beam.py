import state
from mlm_training.model import Model
import numpy as np
import copy

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


    def __len__(self):
        """
        Get the number of states in this beam.
        
        Returns
        =======
        length : int
            The number of states in this beam.
        """
        return len(self.beam)


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



    def add_initial_state(self, model, sess, iterative_pw=False):
        """
        Add an empty initial state to the beam.

        This is used once before the initial beam search begins.

        Parameters
        ==========
        model : Model
            The language model to use for the transduction process

        sess : tf.session
            The tensorflow session of the loaded model.
            
        pitch_wise : boolean
            True to use iterative pitchwise processing (and save only a single hidden_state
            per State). False (default) otherwise.
        """
        if model.pitchwise and not iterative_pw:
            single_state = model.get_initial_state(sess, 1)[0]
            # We have to get 88 initial states, one for each pitch
            initial_state = [copy.copy(single_state) for i in range(88)]
            
        else:
            initial_state = model.get_initial_state(sess, 1)[0]
            
        prior = np.ones(88) / 2 if not iterative_pw else np.array([0.5])

        new_state = state.State()
        new_state.update_from_lstm(initial_state, prior)
        
        self.beam.append(new_state)


        
    def cut_to_size(self, beam_size, hash_length):
        """
        Removes all but the beam_size most probable states from this beam.

        Parameters
        ==========
        beam_size : int
            The maximum number of states to save.

        hash_length : int
            The hash length to save. If two states do not differ in the past hash_length
            frames, only the most probable one is saved in the beam.
        """
        beam = sorted(self.beam, key=lambda s: s.log_prob, reverse=True)
        self.beam = []

        piano_rolls = []

        for state in beam:
            pr = state.get_piano_roll(max_length=hash_length)
            if not any((pr == x).all() for x in piano_rolls):
                self.beam.append(state)
                piano_rolls.append(pr)

                if len(self.beam) == beam_size:
                    break
