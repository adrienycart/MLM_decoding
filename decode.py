import numpy as np
import beam
import itertools
import queue

def decode(acoustic, LSTM, branch_factor=50, beam_size=50):
    """
    Transduce the given acoustic probabilistic piano roll into a binary piano roll.
    
    Parameters
    ==========
    acoustic : matrix
        A probabilistic piano roll, 88 x T, containing values between 0.0 and 1.0
        inclusive. acoustic[p, t] represents the probability of pitch p being present
        at frame t.
        
    LSTM : model
        The language model to use for the transduction process.
        
    branch_factor : int
        The number of samples to use per frame. Defaults to 50.
        
    beam_size : int
        The beam size for the search. Defaults to 50.
        
    
    Returns
    =======
    piano_roll : matrix
        An 88 x T binary piano roll, where a 1 represents the presence of a pitch
        at a given frame.
    """
    beam = beam.Beam()
    beam.add_initial_state()
    
    for frame in np.transpose(acoustic):
        new_beam = beam.Beam()
        
        for state in beam:
            language_prior = get_language_prior(LSTM, state)
            for sample in itertools.islice(enumerate_samples(frame, language_prior), branch_factor):
                log_prob = get_log_prob(sample, language, acoustic)
                new_beam.add(state.get_next_state(sample, log_prob))
                
        new_beam.cut_to_size(beam_size)
        beam = new_beam
        
    return beam.get_top_state().get_piano_roll()



def get_log_prob(sample, acoustic, language):
    """
    Get the log probability of the given sample given the priors.
    
    Parameters
    ==========
    sample : vector
        An 88-length binarized vector, containing pitch detections.
    
    acoustic : vector
        An 88-length array, containing the probability of each pitch being present,
        according to the acoustic model.
        
    language : vector
        An 88-length array, containing the probability of each pitch being present,
        according to the language model.
        
    Returns
    =======
    log_prob : float
        The log probability of the given sample.
    """
    return np.sum(np.where(sample == 1, np.log(language) + np.log(acoustic), np.log(1 - language) + np.log(1 - acoustic)))




def get_language_prior(LSTM, state):
    """
    Get the prior distribution given an LSTM and its current state.
    
    Parameters
    ==========
    LSTM : model
        The LSTM we will use to get the prior distribution.
        
    state : model state
        The state of the LSTM model to use to generate the priors.
        
        
    Returns
    =======
    prior : vector
        An 88-length vector, containing the prior probability for each pitch to be present,
        given the language model and state.
    """
    model = LSTM.load_state(state)
    output = LSTM.evaluate(model)
    return output




def enumerate_samples(acoustic, language, mode="joint"):
    """
    Enumerate the binarised piano-roll samples of a frame, ordered by probability.
    
    Based on Algorithm 2 from:
    Boulanger-Lewandowski, Bengio, Vincent - 2013 - High-dimensional Sequence Transduction
    
    Parameters
    ==========
    acoustic : np.ndarray
        An 88-length array, containing the probability of each pitch being present,
        according to the acoustic model.
        
    language : np.ndarray
        An 88-length array, containing the probability of each pitch being present,
        according to the language model.
        
    mode : string
        How to return the samples. One of "joint" (joint probability), "language"
        (language model only), or "acoustic" (acoustic model only). Defaults to "joint".
        
    Return
    ======
    A generator for the given samples, ordered by probability, which will return
    the log-probability of a sample and the sample itself, a set of indices with an active pitch.
    """
    # set up p and not_p probabilities
    if mode == "joint":
        p = acoustic * language
        not_p = (1 - acoustic) * (1 - language)
        
    elif mode == "language":
        p = language
        not_p = 1 - p
        
    elif mode == "acoustic":
        p = acoustic
        not_p = 1 - p
        
    else:
        raise("Unsupported mode for enumerate_samples: " + str(mode))
    
    # Base case: most likely chosen greedily
    v_0 = np.where(p > not_p)[0]
    l_0 = np.sum(np.log(np.maximum(p, not_p)))
    yield l_0, v_0
    
    # Sort likelihoods by likelihood penalty for flipping
    L = np.abs(np.log(p / not_p))
    R = L.argsort()
    L_sorted = L[R]
    
    # Num solves crash for duplicate likelihoods
    num = 1
    q = queue.PriorityQueue()
    q.put((L_sorted[0], 0, np.array([0])))
    
    while not q.empty():
        l, _, v = q.get()
        yield l_0 - l, np.setxor1d(v_0, R[v])
        
        i = np.max(v)
        if i + 1 < len(L_sorted):
            q.put((l + L_sorted[i + 1], num, np.append(v, i + 1)))
            q.put((l + L_sorted[i + 1] - L_sorted[i], num+1, np.setxor1d(np.append(v, i + 1), [i])))
            num += 2




if __name__ == '__main__':
    pass