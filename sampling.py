import numpy as np
import queue

def enumerate_samples(acoustic, language, mode="joint"):
    """
    Enumerate the binarised piano-roll samples of a frame, ordered by probability.
    
    Based on Algorithm 2 from:
    Boulanger-Lewandowski, Bengio, Vincent - 2013 - High-dimensional Sequence Transduction
    
    Parameters
    ==========
    acoustic : np.ndarray
        An N-length array, containing the probability of each pitch being present,
        according to the acoustic model.
        
    language : np.ndarray
        An N-length array, containing the probability of each pitch being present,
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