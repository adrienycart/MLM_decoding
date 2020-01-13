import numpy as np
import madmom

def sonify_beats(beats,downbeats=None,subbeats=None):
    midi = pm.PrettyMIDI()

    #Add beats
    if not beats is None:
        bell = pm.Instrument(is_drum=True,program=0)
        for beat in beats:
            note_pm = pm.Note(
                velocity=80, pitch=pm.drum_name_to_note_number('Cowbell'), start=beat, end=beat+0.001)
            bell.notes.append(note_pm)
        midi.instruments.append(bell)

    #Add downbeats
    if not downbeats is None:
        triangle = pm.Instrument(is_drum=True,program=0)
        for downbeat in downbeats:
            note_pm = pm.Note(
                velocity=100, pitch=pm.drum_name_to_note_number('Open Triangle'), start=downbeat, end=downbeat+0.001)
            triangle.notes.append(note_pm)
        midi.instruments.append(triangle)

    #Add subbeats
    if not subbeats is None:
        sidestick = pm.Instrument(is_drum=True,program=0)
        for subbeat in subbeats:
            note_pm = pm.Note(
                velocity=60, pitch=pm.drum_name_to_note_number('Side Stick'), start=subbeat, end=subbeat+0.001)
            sidestick.notes.append(note_pm)
        midi.instruments.append(sidestick)

    audio = midi.fluidsynth()

    return audio

def get_subbeat_divisions(beats,beat_activ):
    """
    Compute the number per beat and locations of sub-beat subdivisions.

    Uses :class:`DBNDownBeatTrackingProcessor` from `madmom` library, as in :func:`get_beats_downbeats_signature`.


    Parameters
    ----------
    beats : 1D numpy array
        positions in seconds of the beats
    beat_activ : 1D numpy array
        beat activations

    Returns
    -------
    int
        Number of subdivisions in each beats (2 or 3)
    1D numpy array
        Positions in seconds of the sub-beat subdivisions (only used for visualisation/debugging)
    """

    n_beats = len(beats)
    n_iter=0
    min_bpm = 110.0
    bpm_incr = 55.0
    for i in range(10):
        proc_beat_track = madmom.features.DBNBeatTrackingProcessor(fps=100,min_bpm=min_bpm,max_bpm=600)
        new_beats = proc_beat_track(beat_activ)
        n_new_beats = len(new_beats)
        if n_new_beats!=n_beats:
            #Different beats found, they correspond to sub-beat level
            if abs(round(n_new_beats/2.0) - n_beats) <= 1:
                #If n_new_beats/2.0 is approx equal to n_beats
                return 2, new_beats
            elif abs(round(n_new_beats/3.0) - n_beats) <= 1:
                #If n_new_beats/3.0 is approx equal to n_beats
                return 3, new_beats
            #Default case; it means that the beat found is not an integer subdivision of the original beats
            #Iterate until we find it


        #If function has not returned, iterate with a higher min_bpm
        min_bpm += bpm_incr
    # Consider that default is binary
    return 2, new_beats


def get_confidence_spectral_flatness(activ):
    fs = 100
    window_dur = 10
    overlap = 0.5

    window_len = int(round(window_dur*fs))
    hop = int(round(window_len*(1-overlap)))
    confidence = []
    i = 0
    while i+window_len < len(activ):
        chunk = activ[i:i+window_len]

        spectral_flatness = stats.gmean(chunk) / np.mean(chunk)
        confidence += [spectral_flatness]

        i+=hop

    return confidence

def get_confidence_entropy(activ):
    fs = 100


    window_dur = 12
    step = 0.2

    window_size = int(window_dur*fs)
    step_size = int(step*fs)

    frames = madmom.audio.signal.FramedSignal(activ, frame_size=window_size, hop_size=step_size)
    stft = madmom.audio.stft.ShortTimeFourierTransform(frames,window=np.hanning)
    spec = np.absolute(stft)
    spec_norm = spec/np.sum(spec,axis=1)[:,None]

    entropies = np.sum(-spec_norm*np.log2(spec_norm),axis=1)/np.log2(spec_norm.shape[1])
    return entropies, spec_norm.T
