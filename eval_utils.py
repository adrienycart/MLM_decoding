import numpy as np
import mir_eval

def filter_short_notes(data,thresh=1):
    #Removes all notes shorter than thresh
    #thresh is in number of steps
    data_extended = np.pad(data,((0,0),(0,0),(1,1)),'constant')
    diff = data_extended[:,:,1:] - data_extended[:,:,:-1]

    onsets= np.where(diff==1)
    offsets= np.where(diff==-1)

    mask = offsets[2]-onsets[2]>thresh
    onsets_filt = (onsets[0][mask],onsets[1][mask],onsets[2][mask])
    offsets_filt = (offsets[0][mask],offsets[1][mask],offsets[2][mask])

    diff_filtered=np.zeros(data_extended.shape)

    diff_filtered[onsets_filt]=1
    diff_filtered[offsets_filt]=-1

    return np.cumsum(diff_filtered,axis=2)[:,:,:-2].astype(int)

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
    return np.sum(np.logical_and(data == 1, target == 1),axis=(1,2))

def FP(data,target):
    return np.sum(np.logical_and(data == 1, target == 0),axis=(1,2))

def FN(data,target):
    return np.sum(np.logical_and(data == 0, target == 1),axis=(1,2))

def precision(data,target,mean=True):
    #Compute precision for each file independently (returns a vector of precisions)
    #If mean == True, return the mean across all files
    tp = TP(data,target).astype(float)
    fp = FP(data,target)
    pre_array = tp/(tp+fp+np.full(tp.shape,np.finfo(float).eps))

    if mean:
        return np.mean(pre_array)
    else :
        return pre_array

def recall(data,target,mean=True):
    #Compute recall for each file independently (returns a vector of precisions)
    #If mean == True, return the mean across all files
    tp = TP(data,target).astype(float)
    fn = FN(data,target)
    rec_array = tp/(tp+fn+np.full(tp.shape,np.finfo(float).eps))
    if mean:
        return np.mean(rec_array)
    else :
        return rec_array


def accuracy(data,target,mean=True):
    #Compute accuracy for each file independently (returns a vector of precisions)
    #If mean == True, return the mean across all files
    tp = TP(data,target).astype(float)
    fp = FP(data,target)
    fn = FN(data,target)
    acc_array = tp/(tp+fp+fn+np.full(tp.shape,np.finfo(float).eps))
    if mean :
        return np.mean(acc_array)
    else :
        return acc_array

def Fmeasure(data,target,mean=True):
    #Compute F-measure for each file independently (returns a vector of precisions)
    #If mean == True, return the mean across all files
    prec = precision(data,target,mean=False)
    rec = recall(data,target,mean=False)

    if mean:
        return np.mean(2*prec*rec/(prec+rec+np.full(prec.shape,np.finfo(float).eps)))
    else :
        return 2*prec*rec/(prec+rec+np.full(prec.shape,np.finfo(float).eps))

def compute_eval_metrics_frame(data1,data2,threshold=None):
    #Compute evaluation metrics frame-by-frame
    if not threshold==None:
        idx = data1[:,:,:] > threshold
        data1 = idx.astype(int)


    prec = precision(data1,data2)
    rec = recall(data1,data2)
    # acc = accuracy(data1,data2)
    F = Fmeasure(data1,data2)
    return F, prec, rec

def compute_eval_metrics_note(data1,data2,fs,threshold=None,min_dur=None,tolerance=None):
    #Compute evaluation metrics note-by-note
    #filter out all notes shorter than min_dur (in seconds, default 50ms)
    #A note is correctly detected if it has the right pitch and the inset is within tolerance parameter (default 50ms)
    #Uses the mir_eval implementation
    if not threshold==None:
        idx = data1[:,:,:] > threshold
        data1 = idx.astype(int)

    if min_dur==None:
        data_filt = filter_short_notes(data1,thresh=int(round(fs*0.05)))
    elif min_dur == 0:
        data_filt = data1
    else:
        data_filt = filter_short_notes(data1,thresh=int(round(fs*min_dur)))
    results = []

    if tolerance == None:
        tolerance = 0.05

    for data, target in zip(data_filt,data2):
        notes_est , intervals_est = get_notes_intervals(data, fs)
        notes_ref , intervals_ref = get_notes_intervals(target, fs)

        P,R,F,_ = mir_eval.transcription.precision_recall_f1_overlap(intervals_ref,notes_ref,intervals_est,notes_est,pitch_tolerance=0.25,offset_ratio=None,onset_tolerance=tolerance)
        results += [[F,P,R]]
    results_mean = np.mean(np.array(results),axis=0)
    return results_mean


def get_best_thresh(inputs, targets,lengths,model,save_path,verbose=False,max_thresh=1,step=0.01):
    #Computes on the given dataset the best threshold to use to binarize prediction

    F_list1 = []
    step1 = step*10
    thresh_list1 = np.arange(0,max_thresh,step1)

    for thresh in thresh_list1:
        F, prec, rec, XE = model.compute_eval_metrics_pred(inputs, targets,lengths,threshold=thresh,save_path=save_path)
        F_list1 += [F]
    print thresh_list1
    print F_list1
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
        print "Best F0 : "+str(max_value2)
        print "Best thresh : "+str(max_thresh2)

    return max_thresh2, max_value2

def get_best_eval_metrics(data,model,save_path,fs,verbose=False,quant=False,note_to_time=False,data_time=None,meter=False):
    #Computes the best threshold on the validation dataset,
    #and uses it to return the evaluation metrics on the test dataset.

    inputs, targets, lengths = data.get_dataset('valid',meter=meter)

    thresh,_ = get_best_thresh(inputs, targets,lengths,model,save_path,verbose)


    inputs, targets, lengths = data.get_dataset('test',meter=meter)

    if note_to_time:
        predictions = model.run_prediction(inputs,lengths,save_path,sigmoid=True)
        pred_rolls = (predictions>thresh).astype(int)
        pred_rolls_time = data.convert_note_to_time('test',pred_rolls,100,30)

        _ , target_rolls_time ,_ = data_time.get_dataset('test')

        F_f, prec_f, rec_f = compute_eval_metrics_frame(pred_rolls_time,target_rolls_time)

        F_n, prec_n, rec_n = compute_eval_metrics_note(pred_rolls_time,target_rolls_time,100,min_dur=0)

    else:
        predictions = model.run_prediction(inputs,lengths,save_path,sigmoid=True)
        pred_rolls = (predictions>thresh).astype(int)

        F_f, prec_f, rec_f = compute_eval_metrics_frame(pred_rolls,targets)
        if quant:
            F_n, prec_n, rec_n = compute_eval_metrics_note(pred_rolls,targets,fs,min_dur=0,tolerance=0.001)
        else:
            F_n, prec_n, rec_n = compute_eval_metrics_note(pred_rolls,targets,fs)

    results_f = [F_f, prec_f, rec_f ]
    results_n = [F_n, prec_n, rec_n ]

    if verbose :
        print "Frame  : Fmeasure, precision, recall"
        print results_f
        print "Note  : Fmeasure, precision, recall"
        print results_n

    return [results_f, results_n]

def get_best_eval_metrics_post_quantise(data,model,save_path):
    #Computes the best threshold on the validation dataset,
    #and uses it to return the evaluation metrics on the test dataset,
    #in the case where the computations are made with time-based time steps,
    #and then the onsets and offsets are quantised a posteriori.

    inputs, targets, lengths = data.get_dataset('valid')

    thresh,_ = get_best_thresh(inputs, targets,lengths,model,save_path,True)


    inputs, targets, lengths = data.get_dataset('test')

    predictions = model.run_prediction(inputs,lengths,save_path,sigmoid=True)
    pred_rolls = (predictions>thresh).astype(int)

    #Quantise the onsets and offsets (align them to the 16th note grid)
    pred_rolls_note = data.convert_time_to_note('test',pred_rolls,4,30)
    pred_rolls_time = data.convert_note_to_time('test',pred_rolls_note,100,30)


    F_f, prec_f, rec_f = compute_eval_metrics_frame(pred_rolls_time,targets)
    F_n, prec_n, rec_n = compute_eval_metrics_note(pred_rolls_time,targets,100,min_dur=0)

    results_f = [F_f, prec_f, rec_f ]
    results_n = [F_n, prec_n, rec_n ]


    print "Frame  : Fmeasure, precision, recall"
    print results_f
    print "Note  : Fmeasure, precision, recall"
    print results_n

    return [results_f, results_n]


# from display_utils import compare_piano_rolls
# def get_best_eval_metrics_test(data,targets):
#
#     pred_rolls_note = data.convert_time_to_note(targets[0],4,40)
#     pred_rolls_time = data.convert_note_to_time(pred_rolls_note,100,30)
#     pred_rolls_time = np.asarray([pred_rolls_time])
#
#     compare_piano_rolls([targets[0],pred_rolls_note,pred_rolls_time[0]],[21,109],show=True)
#
#     F_f, prec_f, rec_f = compute_eval_metrics_frame(pred_rolls_time,targets)
#     F_n, prec_n, rec_n = compute_eval_metrics_note(pred_rolls_time,targets,100,min_dur=0)
#
#     results_f = [F_f, prec_f, rec_f ]
#     results_n = [F_n, prec_n, rec_n ]
#
#
#     print "Frame  : Fmeasure, precision, recall"
#     print results_f
#     print "Note  : Fmeasure, precision, recall"
#     print results_n
#
#     return [results_f, results_n]
#
# import pretty_midi as pm
# from dataMaps import DataMaps

# filename = 'data/Config1/fold1/train/MAPS_MUS-deb_clai_ENSTDkCl.mid'
# piano_roll = DataMaps()
# piano_roll.make_from_file(filename,100,quant=False,posteriogram=True)
# # time_grid = [int(round(x*100))-2 for x in piano_roll.corresp[:,0]]
# # print piano_roll.corresp[:,0]
# # print time_grid
#
# compare_piano_rolls([piano_roll.target],[21,109],time_grid=time_grid,show=True)



# piano_roll_target = np.array([(pm.PrettyMIDI(filename).get_piano_roll()>0)[21:109,:3000].astype(int)])
# get_best_eval_metrics_test(piano_roll,piano_roll_target)
