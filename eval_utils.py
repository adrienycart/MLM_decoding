import numpy as np
import mir_eval
import pretty_midi as pm

def filter_short_notes_roll(data,thresh=1):
    #Removes all notes shorter than thresh
    #thresh is in number of steps
    data_extended = np.pad(data,((0,0),(1,1)),'constant')
    diff = data_extended[:,1:] - data_extended[:,:-1]

    onsets= np.where(diff==1)
    offsets= np.where(diff==-1)

    mask = offsets[1]-onsets[1]>thresh
    onsets_filt = (onsets[0][mask],onsets[1][mask])
    offsets_filt = (offsets[0][mask],offsets[1][mask])

    diff_filtered=np.zeros(data_extended.shape)

    diff_filtered[onsets_filt]=1
    diff_filtered[offsets_filt]=-1

    return np.cumsum(diff_filtered,axis=1)[:,:-2].astype(int)


def filter_short_gaps_roll(data,thresh=1):
    #Removes all gaps shorter than thresh
    #thresh is in number of steps

    data = 1 - data
    data_filt = filter_short_notes_roll(data,thresh)
    data_filt = 1-data_filt

    return data_filt

def filter_short_notes(pitches,intervals,thresh=0.05):
    to_keep = (intervals[:,1]-intervals[:,0])>=thresh
    return pitches[to_keep], intervals[to_keep]

def filter_short_gaps(pitches,intervals,thresh=0.05):

    # TODO: CURRENTLY DOESN'T WORK!!!

    intervals_array = np.array([128],dtype=object)
    for i in range(128):
        intervals_array[i] = []

    # First, fill intervals_array
    for pitch, [onset,offset] in zip(pitches,intervals):
        intervals_array[pitch] += [[onset,offset]]


    for i in range(128):
        # Sort all intervals per onset time (for each pitch)
        sorted_intervals = sorted(intervals_array[i],key = lambda x: x[0])
        # Then, for each pitch, remove short gaps
        for j in range(len(sorted_intervals)-1):
            assert sorted_intervals[j][1]<=sorted_intervals[j+1][0]



def get_notes_intervals(pr,fs):
    #Returns the list of note events from a piano-roll

    data_extended = np.pad(pr,((0,0),(1,1)),'constant')
    diff = data_extended[:,1:] - data_extended[:,:-1]

    #Onset: when a new note activates (doesn't count repeated notes)
    onsets= np.where(diff==1)
    #Onset: when a new note deactivates (doesn't count repeated notes)
    offsets= np.where(diff==-1)

    assert onsets[0].shape == offsets[0].shape
    assert onsets[1].shape == offsets[1].shape

    pitches = []
    intervals = []
    for [pitch1,onset], [pitch2,offset] in zip(zip(onsets[0],onsets[1]),zip(offsets[0],offsets[1])):
        # print pitch1, pitch2
        # print onset, offset
        assert pitch1 == pitch2
        # Add +1 because pitches cannot be equal to zeros for evaluation
        pitches += [pitch1+1]
        if fs is None:
            intervals += [[onset, offset]]
        else:
            intervals += [[onset/float(fs), offset/float(fs)]]
        # print pitches
        # print intervals
    return np.array(pitches), np.array(intervals)


def get_notes_intervals_with_onsets(pr,corresp,double_roll=False,add_missing_onsets=False,merge_consecutive_onsets=False):
    #Returns the list of note events from a piano-roll

    if double_roll:
        onsets_matrix = pr[int(pr.shape[0]/2):,:]
        note_on_matrix = pr[:int(pr.shape[0]/2),:]
    else:
        onsets_matrix = (pr==2).astype(int)
        note_on_matrix = (pr==1).astype(int)

    if add_missing_onsets:
        # Whenever there is a note_on without an onset either at the
        # same timestep (if double_roll is True) or just before (if double_roll
        # is false), we add an extra onset
        data_extended = np.pad(note_on_matrix,((0,0),(1,1)),'constant')
        diff = data_extended[:,1:] - data_extended[:,:-1]

        #Onset: when a new note activates (doesn't count repeated notes)
        onsets= np.where(diff==1)
        for pitch, onset in zip(onsets[0],onsets[1]):
            if double_roll:
                if onsets_matrix[pitch,onset]==0:
                    onsets_matrix[pitch,onset] = 1
            else:
                if onsets_matrix[pitch,onset-1]==0:
                    onsets_matrix[pitch,onset-1] = 1

    if merge_consecutive_onsets:
        data_extended = np.pad(onsets_matrix,((0,0),(1,1)),'constant')
        diff = data_extended[:,1:] - data_extended[:,:-1]
        mask = np.logical_and(diff[:,:-1]==0,onsets_matrix==1)
        onsets_matrix[mask]=0



    # Only gather notes that have an onset
    onsets= np.where(onsets_matrix==1)

    pitches = []
    intervals = []
    for pitch, onset in zip(onsets[0],onsets[1]):
        if onset == pr.shape[1]-1:
            offset=onset+1
        else:
            # Offset is when note_on goes off, or when there is an onset, whatever happens first
            offset_array = np.logical_and(note_on_matrix[pitch,onset+1:],1-onsets_matrix[pitch,onset+1:]).astype(int)
            if np.all(offset_array):
                # Only ones in offset_array, offset is the end of the array
                offset = pr.shape[1]
            else:
                dur = np.argmin(offset_array) # Argmin returns index of first zero
                # +1 because offset_array starts at onset+1
                offset = dur+onset+1
        # Add +1 because pitches cannot be equal to zeros for evaluation
        pitches += [pitch+1]
        intervals += [[corresp[onset],corresp[offset]]]

    return np.array(pitches), np.array(intervals)


def TP(data,target):
    return np.sum(np.logical_and(data == 1, target == 1))

def FP(data,target):
    return np.sum(np.logical_and(data == 1, target == 0))

def FN(data,target):
    return np.sum(np.logical_and(data == 0, target == 1))

def precision(data,target):
    #Compute precision for  one file
    tp = TP(data,target).astype(float)
    fp = FP(data,target)
    pre = tp/(tp+fp+np.finfo(float).eps)

    return pre

def recall(data,target):
    #Compute recall for  one file
    tp = TP(data,target).astype(float)
    fn = FN(data,target)
    rec = tp/(tp+fn+np.finfo(float).eps)
    return rec


def accuracy(data,target):
    #Compute accuracy for one file
    tp = TP(data,target).astype(float)
    fp = FP(data,target)
    fn = FN(data,target)
    acc = tp/(tp+fp+fn+np.finfo(float).eps)
    return acc

def Fmeasure(data,target):
    #Compute F-measure  one file
    prec = precision(data,target)
    rec = recall(data,target)
    return 2*prec*rec/(prec+rec+np.finfo(float).eps)

def compute_eval_metrics_frame(input,target):
    #Compute evaluation metrics frame-by-frame
    input = input[:,:min(input.shape[1], target.shape[1])]
    target = target[:,:min(input.shape[1], target.shape[1])]

    prec = precision(input,target)
    rec = recall(input,target)
    # acc = accuracy(input,target)
    F = Fmeasure(input,target)
    return prec, rec, F

def compute_eval_metrics_note(input,target,min_dur=None,tolerance=None, with_offset=False, min_gap=None,merge_consecutive_onsets=False):
    #Compute evaluation metrics note-by-note
    #filter out all notes shorter than min_dur (in seconds, default 50ms)
    #A note is correctly detected if it has the right pitch and the inset is within tolerance parameter (default 50ms)
    #Uses the mir_eval implementation

    #All inputs should be with 40ms timesteps
    fs = 25



    if min_dur==None:
        data_filt = filter_short_notes_roll(input,thresh=int(round(fs*0.05)))
    elif min_dur == 0:
        data_filt = input
    else:
        data_filt = filter_short_notes_roll(input,thresh=int(round(fs*min_dur)))

    if min_gap is not None:
        data_filt = filter_short_gaps(input,thresh=int(round(fs*min_gap)))


    if tolerance == None:
        tolerance = 0.05

    if with_offset:
        offset_ratio = 0.2
    else:
        offset_ratio = None


    notes_est , intervals_est = get_notes_intervals(input, fs)
    notes_ref , intervals_ref = get_notes_intervals(target, fs)

    if len(notes_est) == 0:
        return 0,0,0
    else:
        P,R,F,_ = mir_eval.transcription.precision_recall_f1_overlap(intervals_ref,
        notes_ref,intervals_est,notes_est,pitch_tolerance=0.25,offset_ratio=offset_ratio,
        onset_tolerance=tolerance,offset_min_tolerance=0.05)
        return P,R,F

def compute_eval_metrics_with_onset(input_pr,corresp,target_data,section=None,double_roll=False,min_dur=None,tolerance=None, with_offset=False, min_gap=None,merge_consecutive_onsets=False):
    #Compute evaluation metrics note-by-note
    #filter out all notes shorter than min_dur (in seconds, default 50ms)
    #A note is correctly detected if it has the right pitch and the inset is within tolerance parameter (default 50ms)
    #Uses the mir_eval implementation


    # Get note sequences
    notes_est , intervals_est = get_notes_intervals_with_onsets(input_pr, corresp,double_roll,merge_consecutive_onsets=merge_consecutive_onsets)
     #Note +20 and not 21 because get_notes_intervals adds +1
    notes_est = notes_est+20
    if len(intervals_est) == 0:
        intervals_est = np.zeros((0, 2))



    notes_ref,intervals_ref = [],[]

    for note in sum([instr.notes for instr in target_data.instruments],[]):
        if section is None or (note.start < section[1] and note.end>section[0]):
            notes_ref+= [note.pitch]
            intervals_ref+= [[max(note.start,section[0]),min(note.end,section[1])]]

    notes_ref = np.array(notes_ref)
    intervals_ref = np.array(intervals_ref)


    if min_dur==None:
        notes_est , intervals_est = filter_short_notes(notes_est, intervals_est,0.05)
    elif min_dur == 0:
        pass
    else:
        notes_est , intervals_est = filter_short_notes(notes_est , intervals_est,thresh=min_dur)

    if min_gap is not None:
        # No min_gap filtering
        print("No min_gap filtering with onsets!")
        # data_filt = filter_short_gaps(input,thresh=int(round(fs*min_gap)))

    # Get rolls
    fs = 100
    target = (target_data.get_piano_roll(fs=100)>0).astype(int)
    if section is not None:
        target = target[:,int(section[0]*fs):int(section[1]*fs)]
    output = np.zeros_like(target)
    for pitch, [onset,offset] in zip(notes_est,intervals_est):
        on_idx = int(onset*fs)
        off_idx = int(offset*fs)
        output[pitch,on_idx:off_idx]=1


    if tolerance == None:
        tolerance = 0.05

    if with_offset:
        offset_ratio = 0.2
    else:
        offset_ratio = None


    P_f = precision(output,target)
    R_f = recall(output,target)
    F_f = Fmeasure(output,target)


    if len(notes_est) == 0:
        P_n,R_n,F_n = 0,0,0
    else:
        P_n,R_n,F_n,_ = mir_eval.transcription.precision_recall_f1_overlap(intervals_ref,
        notes_ref,intervals_est,notes_est,pitch_tolerance=0.25,offset_ratio=offset_ratio,
        onset_tolerance=tolerance,offset_min_tolerance=0.05)

    return [P_f,R_f,F_f],[P_n,R_n,F_n], notes_est, intervals_est

def out_key_errors_binary_mask(input,target,mask,mask_octave,min_dur=None,tolerance=None, with_offset=False, min_gap=None, mask_thresh=0.05):
    #Compute evaluation metrics note-by-note
    #filter out all notes shorter than min_dur (in seconds, default 50ms)
    #A note is correctly detected if it has the right pitch and the inset is within tolerance parameter (default 50ms)
    #Uses the mir_eval implementation

    #All inputs should be with 40ms timesteps
    fs = 25


    if min_dur==None:
        data_filt = filter_short_notes_roll(input,thresh=int(round(fs*0.05)))
    elif min_dur == 0:
        data_filt = input
    else:
        data_filt = filter_short_notes_roll(input,thresh=int(round(fs*min_dur)))

    if min_gap is not None:
        data_filt = filter_short_gaps(input,thresh=int(round(fs*min_gap)))

    results = []

    if tolerance == None:
        tolerance = 0.05

    if with_offset:
        offset_ratio = 0.2
    else:
        offset_ratio = None


    notes_est , intervals_est = get_notes_intervals(input, fs)
    notes_ref , intervals_ref = get_notes_intervals(target, fs)

    match = mir_eval.transcription.match_notes(intervals_ref, notes_ref, intervals_est, notes_est,offset_ratio=None, pitch_tolerance=0.25)



    in_mask = mask>mask_thresh
    in_mask_octave = mask_octave>mask_thresh


    # import matplotlib.pyplot as plt
    # fig, (ax0,ax1) = plt.subplots(2,1)
    # ax0.plot(mask_octave)
    # ax1.plot(in_mask_octave.astype(int))
    # plt.show(block=True)


    if len(match) == 0:
        unmatched_outputs = list(range(len(notes_est)))
    else:
        matched_targets, matched_outputs = zip(*match)
        unmatched_outputs= list(set(range(len(notes_est)))-set(matched_outputs))

    if len(unmatched_outputs) == 0:
        return 0.0,0.0
    else:
        out_key_unmatched = []
        out_key_unmatched_octave = []
        for i in unmatched_outputs:
            # print(in_mask[notes_est[i]])
            if not in_mask[notes_est[i]-1]: #-1 because we add +1 in get_notes_intervals
                out_key_unmatched += [notes_est[i]]

            if not in_mask_octave[(notes_est[i]-1)%12]:
                out_key_unmatched_octave += [notes_est[i]]

        tot_out_key = float(len(out_key_unmatched))
        tot_out_key_o = float(len(out_key_unmatched_octave))
        tot_err = len(unmatched_outputs)
        tot_notes = len(notes_est)

        return tot_out_key/tot_err, tot_out_key/tot_notes,tot_out_key_o/tot_err, tot_out_key_o/tot_notes


def get_best_thresh(inputs, targets,lengths,model,save_path,verbose=False,max_thresh=1,step=0.01):
    #Computes on the given dataset the best threshold to use to binarize prediction

    F_list1 = []
    step1 = step*10
    thresh_list1 = np.arange(0,max_thresh,step1)

    for thresh in thresh_list1:
        F, prec, rec, XE = model.compute_eval_metrics_pred(inputs, targets,lengths,threshold=thresh,save_path=save_path)
        F_list1 += [F]
    print(thresh_list1)
    print(F_list1)
    max_value1 = max(F_list1)
    max_index1 = F_list1.index(max_value1)
    max_thresh1 = thresh_list1[max_index1]

    F_list2 = []
    thresh_list2 = np.arange(max(0,max_thresh1-(step1-step)),min(max_thresh,max_thresh1+(step1+step+step/2.0)),step)
    for thresh in thresh_list2:
        F, prec, rec, XE = model.compute_eval_metrics_pred(inputs, targets,lengths,threshold=thresh,save_path=save_path)
        F_list2 += [F]

    max_value2 = max(F_list2)
    max_index2 = F_list2.index(max_value2)
    max_thresh2 = thresh_list2[max_index2]

    if verbose:
        model.print_params()
        print("Best F0 : "+str(max_value2))
        print("Best thresh : "+str(max_thresh2))

    return max_thresh2, max_value2


#####################################################
#### To synthesize some pianorolls
#####################################################

def make_midi_from_roll(roll,fs):
    #Outputs the waveform corresponding to the pianoroll

    pitches, intervals = get_notes_intervals(roll,fs)
    pitches = pitches+20 #Note +20 and not 21 because get_notes_intervals adds +1
    return make_midi_from_notes(pitches,intervals)


def make_midi_from_notes(notes,intervals):
    midi_data = pm.PrettyMIDI(resolution=480)
    piano_program = pm.instrument_name_to_program('Acoustic Grand Piano')
    piano = pm.Instrument(program=piano_program)

    for note,(start,end) in zip(notes,intervals):
        note = pm.Note(
            velocity=100, pitch=note, start=start, end=end)
        piano.notes.append(note)
    midi_data.instruments.append(piano)
    return midi_data


def save_midi(midi,dest):
    midi.write(dest)

def synthesize_midi(midi,dest):
    # Requires fluidsynth and pyFluidSynth installed!!!
    return midi.fluidsynth()

def write_sound(sound,filename):
    sound = 16000*sound #increase gain
    wave_write = wave.open(filename,'w')
    wave_write.setparams([1,2,44100,10,'NONE','noncompressed'])
    ssignal = ''
    for i in range(len(sound)):
       ssignal += wave.struct.pack('h',sound[i]) # transform to binary
    wave_write.writeframes(ssignal)
    wave_write.close()


def play_audio(audio,fs=44100,from_sec=0):
    """
    Play some audio. Requires the :mod:`sounddevice` module.
    Audio must be sampled at 44100 Hz.
    ctrl-C stops the sound and continues the script.

    Parameters
    ----------
    audio: numpy array
        Audio samples
    from_sec: float
        Play start position in seconds
    """

    start_sample = int(round(from_sec*44100))
    import sounddevice as sd
    try:
        sd.play(sig, fs)
        status = sd.wait()
    except KeyboardInterrupt:
        sd.stop()
        return

def mix_sounds(sig1,sig2,sig_mix_ratio=0.5):
    len1 = sig1.shape[0]
    len2 = sig2.shape[0]
    if len1 > len2:
        audio = (1-sig_mix_ratio)*sig1 + sig_mix_ratio*np.pad(sig2,(0,len1-len2),'constant')
    else:
        audio = (1-sig_mix_ratio)*np.pad(sig1,(0,len2-len1),'constant') + sig_mix_ratio*sig2
    return audio
