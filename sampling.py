import numpy as np
import bisect
import queue



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
        A list of the indices where there should be 1's in the samples, transformed using
        the given transform function, if not None.
    """
    length = len(p)
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


def trinarize_with_onsets(sample, P):
    """
    Trinarize the given sample.
    
    Parameters
    ----------
    sample : list(int)
        A list of indices where there should be a sample.
        
    P : int
        The desired length of the resulting sample.
        
        
    Returns
    -------
    trinary_sample : list(int)
        A trinary vector, with 1 indicating presence, and 2 indicating onset.
    """
    num_pitches = P // 2
    
    # Starting index of onsets
    index = bisect.bisect_left(sample, num_pitches)
    
    # Make trinary sample
    trinary_sample = np.zeros(num_pitches)
    trinary_sample[sample[:index]] = 1
    trinary_sample[np.array(sample[index:], dtype=int) - num_pitches] = 2
    
    return trinary_sample
    

def binarize(sample, P):
    """
    Binarize the given sample.
    
    Parameters
    ----------
    sample : list(int)
        A list of indices where there should be a sample.
        
    P : int
        The desired length of the resulting sample.
        
        
    Returns
    -------
    binary_sample : list(int)
        A binary vector with 1 in the indices given by the sample.
    """
    binary_sample = np.zeros(P)
    binary_sample[sample] = 1
    
    return binary_sample