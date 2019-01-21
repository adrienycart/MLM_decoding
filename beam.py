import state

class Beam:
    """
    The beam for performing beam search.
    
    This beam stores the current states and has the ability to cut it down
    to a given size, which will save only the most-probable N states.
    """
    
    def __init__self(self, beam_size):
        """
        Create a new beam, initially empty.
        """
        self.beam = []
    
    
    def add_initial_state():
        """
        Add an empty initial state to the beam.
        
        This is used once before the initial beam search begins.
        """
        self.beam.append(state.State())
            
        
    def cut_to_size(beam_size):
        """
        Removes all but the beam_size most probable states from this beam.
        
        Parameters
        ==========
        beam_size : int
            The maximum number of states to save.
        """
        self.beam = sorted(self.beam, reverse=True)[beam_size]
