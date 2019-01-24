import numpy as np
import mir_eval

def filter_short_notes(data,thresh=1):
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


def get_notes_intervals(data,fs):
    #Returns the list of note events from a piano-roll

    data_extended = np.pad(data,((0,0),(1,1)),'constant')
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
        pitches += [pitch1+1]
        if fs is None:
            intervals += [[onset, offset]]
        else:
            intervals += [[onset/float(fs), offset/float(fs)]]
        # print pitches
        # print intervals
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

    prec = precision(input,target)
    rec = recall(input,target)
    # acc = accuracy(input,target)
    F = Fmeasure(input,target)
    return prec, rec, F

def compute_eval_metrics_note(input,target,min_dur=None,tolerance=None):
    #Compute evaluation metrics note-by-note
    #filter out all notes shorter than min_dur (in seconds, default 50ms)
    #A note is correctly detected if it has the right pitch and the inset is within tolerance parameter (default 50ms)
    #Uses the mir_eval implementation

    #All inputs should be with 40ms timesteps
    fs = 25


    if min_dur==None:
        data_filt = filter_short_notes(input,thresh=int(round(fs*0.05)))
    elif min_dur == 0:
        data_filt = input
    else:
        data_filt = filter_short_notes(input,thresh=int(round(fs*min_dur)))
    results = []

    if tolerance == None:
        tolerance = 0.05


    notes_est , intervals_est = get_notes_intervals(input, fs)
    notes_ref , intervals_ref = get_notes_intervals(target, fs)

    P,R,F,_ = mir_eval.transcription.precision_recall_f1_overlap(intervals_ref,notes_ref,intervals_est,notes_est,pitch_tolerance=0.25,offset_ratio=None,onset_tolerance=tolerance)
    return P,R,F


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
