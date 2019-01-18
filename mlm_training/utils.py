from random import shuffle
import os
import pretty_midi as pm
import numpy as np
import mir_eval.transcription


def safe_mkdir(dir,clean=False):
    if not os.path.exists(dir):
        os.makedirs(dir)
    if clean and not os.listdir(dir) == [] :
        old_path = os.path.join(dir,"old")
        safe_mkdir(old_path)
        for fn in os.listdir(dir):
            full_path = os.path.join(dir,fn)
            if not os.path.isdir(full_path):
                os.rename(full_path,os.path.join(old_path,fn))

def move_files(file_list,folder1,folder2):
    for midi_file in file_list:
        os.rename(os.path.join(folder1,midi_file), os.path.join(folder2,midi_file))

def split_files(folder,test=0.2,valid=0.1):

    midi_list = [x for x in os.listdir(folder) if x.endswith('.mid')]

    train_path = os.path.join(folder,"train/")
    valid_path = os.path.join(folder,"valid/")
    test_path = os.path.join(folder,"test/")

    safe_mkdir(train_path)
    safe_mkdir(valid_path)
    safe_mkdir(test_path)

    N = len(midi_list)
    N_test = int(N*test)
    N_valid = int(N*valid)

    shuffle(midi_list)
    test_list, valid_list, train_list = midi_list[:N_test], midi_list[N_test:N_test+N_valid],midi_list[N_test+N_valid:]

    move_files(test_list,folder,test_path)
    move_files(valid_list,folder,valid_path)
    move_files(train_list,folder,train_path)

def unsplit_files(folder):
    train_path = os.path.join(folder,"train/")
    valid_path = os.path.join(folder,"valid/")
    test_path = os.path.join(folder,"test/")

    move_files(os.listdir(train_path),train_path,folder)
    move_files(os.listdir(valid_path),valid_path,folder)
    move_files(os.listdir(test_path),test_path,folder)

    os.rmdir(train_path)
    os.rmdir(test_path)
    os.rmdir(valid_path)

def write_namelist_as_pickle(folder):
    #Only used once, to make sure the local and distant datasets are the same
    def get_name_list(subfolder):
        namelist = []
        for fn in os.listdir(subfolder):
            if fn.endswith('.mid'):
                namelist += [fn]
        return namelist

    train_path = os.path.join(folder,'train')
    valid_path = os.path.join(folder,'valid')
    test_path = os.path.join(folder,'test')

    namelist = {'train': get_name_list(train_path),
                 'valid':get_name_list(valid_path),
                 'test':get_name_list(test_path)}

    import pickle as pickle
    pickle.dump(namelist, open(os.path.join(folder,'namelist.p'), "wb"))

def split_files_with_namelist(folder):
    #Only used once, to make sure the local and distant datasets are the same
    import pickle as pickle
    namelist = pickle.load(open(os.path.join(folder,'namelist.p'), "rb"))

    train_list = namelist['train']
    valid_list = namelist['valid']
    test_list = namelist['test']

    train_path = os.path.join(folder,'train')
    valid_path = os.path.join(folder,'valid')
    test_path = os.path.join(folder,'test')

    safe_mkdir(train_path)
    safe_mkdir(valid_path)
    safe_mkdir(test_path)

    move_files(test_list,folder,test_path)
    move_files(valid_list,folder,valid_path)
    move_files(train_list,folder,train_path)

def get_chord_counter(subfolder):
    from collections import Counter

    midi_list = [x for x in os.listdir(subfolder) if x.endswith('.mid')]
    chord_list = []
    for midi in midi_list:
        chords = midi.split("_")
        suffix = chords.pop()
        if suffix == "0.mid" or len(chords) == 1:
            chords = [chords[0],chords[0],chords[0]]
        elif suffix == "01.mid":
            chords = [chords[0],chords[1],chords[1]]
        elif suffix == "02.mid":
            chords = [chords[0],chords[0],chords[1]]
        chord_list += chords
    counter = Counter(chord_list)
    return sorted(counter.items())

def get_chord_counter_by_position(subfolder):
    from collections import Counter

    midi_list = [x for x in os.listdir(subfolder) if x.endswith('.mid')]
    chord_list = []
    for midi in midi_list:
        chords = midi.split("_")
        suffix = chords.pop()
        if suffix == "0.mid" or len(chords) == 1:
            chords = [chords[0],chords[0],chords[0]]
        elif suffix == "01.mid":
            chords = [chords[0],chords[1],chords[1]]
        elif suffix == "02.mid":
            chords = [chords[0],chords[0],chords[1]]
        elif suffix == "012.mid":
            print(chords)
        chord_list += [chords]

    def count_position(i):
        count_list = []
        for chord in chord_list:
            count_list += [chord[i]]
        return Counter(count_list)

    counter0 = count_position(0)
    counter1 = count_position(1)
    counter2 = count_position(2)
    return sorted(counter0.items()),sorted(counter1.items()),sorted(counter2.items())

def my_get_end_time(midi_data):
    instruments = midi_data.instruments
    events = []
    for instr in instruments:
        events += [n.end for n in instr.notes]
    # If there are no events, just return 0
    if len(events) == 0:
        return 0.
    else:
        return max(events)

def check_corrupt(subfolder):
    midi_list = [os.path.join(subfolder,x) for x in os.listdir(subfolder) if x.endswith('.mid')]
    for midi_file in midi_list:
        midi = pm.PrettyMIDI(midi_file)
        piano_roll = midi.get_piano_roll()
        len1 = piano_roll.shape[1]/100
        len2 = my_get_end_time(midi)
        if abs(len1-len2) > 1:
            print(midi_file+", len1 = "+str(len1)+", len2 = "+str(len2))


def filter_short_notes(data,thresh=1):
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
    data_extended = np.pad(data,((0,0),(1,1)),'constant')
    diff = data_extended[:,1:] - data_extended[:,:-1]

    onsets= np.where(diff==1)
    offsets= np.where(diff==-1)

    assert onsets[0].shape == offsets[0].shape
    assert onsets[1].shape == offsets[1].shape



    pitches = []
    intervals = []
    for [pitch1,onset], [pitch2,offset] in zip(list(zip(onsets[0],onsets[1])),list(zip(offsets[0],offsets[1]))):
        # print pitch1, pitch2
        # print onset, offset
        assert pitch1 == pitch2
        pitches += [pitch1+1]
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

    tp = TP(data,target).astype(float)
    fp = FP(data,target)
    pre_array = tp/(tp+fp+np.full(tp.shape,np.finfo(float).eps))

    if mean:
        return np.mean(pre_array)
    else :
        return pre_array

def recall(data,target,mean=True):
    tp = TP(data,target).astype(float)
    fn = FN(data,target)
    rec_array = tp/(tp+fn+np.full(tp.shape,np.finfo(float).eps))
    if mean:
        return np.mean(rec_array)
    else :
        return rec_array


def accuracy(data,target,mean=True):
    tp = TP(data,target).astype(float)
    fp = FP(data,target)
    fn = FN(data,target)
    acc_array = tp/(tp+fp+fn+np.full(tp.shape,np.finfo(float).eps))
    if mean :
        return np.mean(acc_array)
    else :
        return acc_array

def Fmeasure(data,target,mean=True):
    prec = precision(data,target,mean=False)
    rec = recall(data,target,mean=False)

    if mean:
        return np.mean(2*prec*rec/(prec+rec+np.full(prec.shape,np.finfo(float).eps)))
    else :
        return 2*prec*rec/(prec+rec+np.full(prec.shape,np.finfo(float).eps))

def compute_eval_metrics_frame(data1,data2,threshold=None):
    if not threshold==None:
        idx = data1[:,:,:] > threshold
        data1 = idx.astype(int)


    prec = precision(data1,data2)
    rec = recall(data1,data2)
    # acc = accuracy(data1,data2)
    F = Fmeasure(data1,data2)
    return F, prec, rec

def compute_eval_metrics_note(data1,data2,fs,threshold=None,min_dur=None,tolerance=None):
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
        # print notes_est.shape
        # print intervals_est.shape
        notes_ref , intervals_ref = get_notes_intervals(target, fs)

        # pairs = mir_eval.transcription.match_notes(intervals_ref, notes_ref, intervals_est, notes_est, onset_tolerance=0.8, pitch_tolerance=0.25, offset_ratio=None)
        # for i,j in pairs:
        #     print ""
        #     print notes_ref[i], intervals_ref[i]
        #     print notes_est[j], intervals_est[j]



        P,R,F,_ = mir_eval.transcription.precision_recall_f1_overlap(intervals_ref,notes_ref,intervals_est,notes_est,pitch_tolerance=0.25,offset_ratio=None,onset_tolerance=tolerance)
        results += [[F,P,R]]
    results_mean = np.mean(np.array(results),axis=0)
    return results_mean


def get_best_thresh(inputs, targets,verbose=False):

    F_list1 = []
    thresh_list1 = np.arange(0,1,0.1)

    for thresh in thresh_list1:
        inputs_thresh = (inputs>thresh).astype(int)
        F = Fmeasure(inputs_thresh, targets)
        F_list1 += [F]

    max_value1 = max(F_list1)
    max_index1 = F_list1.index(max_value1)
    max_thresh1 = thresh_list1[max_index1]

    F_list2 = []
    thresh_list2 = np.arange(max(0,max_thresh1-0.09),min(1,max_thresh1+0.095),0.01)
    for thresh in thresh_list2:
        inputs_thresh = (inputs>thresh).astype(int)
        F = Fmeasure(inputs_thresh, targets)
        F_list2 += [F]

    max_value2 = max(F_list2)
    max_index2 = F_list2.index(max_value2)
    max_thresh2 = thresh_list2[max_index2]

    if verbose:
        print("Best F0 : "+str(max_value2))
        print("Best thresh : "+str(max_thresh2))

    return max_thresh2, max_value2
