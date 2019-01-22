# -*- coding: utf-8 -*-

import pretty_midi as pm
import numpy as np
import os
import copy




pm.pretty_midi.MAX_TICK = 1e10

class DataMaps:
    """Classe representing a couple (posteriogram, ground truth)."""

    def __init__(self):
        self.input = [] # posteriogram
        self.target = [] # transcription ground-truth
        self.corresp = [] # correspondance table, used to convert time-based to note-based time steps
        self.meter_grid = [] # Meter grid
        self.keys = [] # list of couples (key, time), where key is an integer between 0 and 23 and time is in seconds
        self.sigs = [] #list of couples (sig, time), where sig is a list of size 2 and time is in seconds
        self.name = ""
        self.length = 0 # length of the piece (might be different from len(input) because of zero padding)
        self.note_range=[0,128]
        self.begin=0 # beginning of the considered section in seconds
        self.end=0 # end of the considered section in seconds
        self.transp = 0 # transposition of the target in semitones (when using data augmentation)
        self.quant = False # True only when using note-based time steps
        self.posteriogram = False # True iff the input is a posteriogram (as opposed to a piano-roll)

    def make_from_file(self,filename,fs,section=None,note_range=[21,109],quant=False,posteriogram=False,method='avg',transp=None,annot=None):
        self.quant = quant
        self.posteriogram = posteriogram
        pm_data, csv_matrix, corresp, thresh = self.get_pm_and_csv(filename,quant,posteriogram,transp,annot)
        self.make_from_pm_and_csv(pm_data,csv_matrix,corresp,thresh,fs,section,note_range,quant,posteriogram,method,transp)
        if not annot is None:
            self.set_sigs_and_keys(annot)
            self.set_meter_grid(fs)
        return

    def get_pm_and_csv(self,filename,quant,posteriogram,transp=None,annot=None):
        #Method to get a pretty_midi object and an input matrix from a MIDI filename.
        #Also returns the corresp table and the threshold used for note-based time steps
        self.set_name_from_maps(filename)
        if posteriogram :
            post_suffix = '_posteriogram.csv'
        else:
            post_suffix = '_pianoroll.csv'

        if transp == None:
            transp_suffix = ''
        else:
            if transp==0:
                transp_suffix = '_0'
            elif transp > 0:
                transp_suffix = '_+'+str(transp)
            elif transp < 0:
                transp_suffix = '_'+str(transp)
        csv_filename = filename.replace('.mid',transp_suffix+post_suffix)

        # print "csv", csv_filename

        if quant:
            csv_matrix = np.transpose(import_csv(csv_filename),[1,0])
            length = csv_matrix.shape[1]/100.0
            corresp, thresh = self.make_corresp_table(filename,length_time=length,annot=annot)
            self.corresp=corresp
            midi_data = pm.PrettyMIDI(filename)
            return midi_data, csv_matrix, corresp, thresh
        else:
            #ONLY USEFUL FOR TIME-TO-NOTES CONVERSION FOR POST-QUANTISATION
            corresp, thresh = self.make_corresp_table(filename,length_time=None)
            self.corresp=corresp
            #

            csv_matrix = np.transpose(import_csv(csv_filename),[1,0])
            midi_data = pm.PrettyMIDI(filename)
            return midi_data, csv_matrix, None, None #corresp and thresh are not used when not quant

    def set_transp(self,filename):
        suffix = (os.path.splitext(filename)[0]).split('_')[-1]
        try :
            transp = int(suffix)
        except ValueError:
            transp=0
        self.transp=transp
        return

    def make_from_pm_and_csv(self,pm_data,csv_matrix,corresp,thresh,fs,section=None,note_range=[21,109],quant=False,posteriogram=False,method='avg',transp=None):
        # self.note_range = note_range
        if quant:
            self.target = self.get_aligned_pianoroll(pm_data,corresp,section)
            self.input = self.get_aligned_input(csv_matrix,corresp,section,method)
            self.quant = True
            #Target has to be binarized because the pretty_midi get_piano_roll function
            #returns a real-value piano-roll.
            #The threshold was pre-computed to give the best matching between Piano-Midi.de and MAPS files.

            self.binarize_target(thresh)


        else:
            piano_roll = pm_data.get_piano_roll(fs)
            if not section == None :
                #Select the relevant portion of the piano-roll and input
                min_time_step = int(round(section[0]*fs))
                max_time_step = int(round(section[1]*fs))
                self.target = piano_roll[:,min_time_step:max_time_step]
                self.input = csv_matrix[:,min_time_step:max_time_step]
                self.begin = section[0]
                self.end = min(section[1],piano_roll.shape[1]/100.0)
            else:
                #Keep the whole piano-roll
                self.target = piano_roll
                self.input = csv_matrix
                self.begin = 0
                self.end = piano_roll.shape[1]/100.0

            self.quant = False
            self.binarize_target(5)

        if not self.transp==0:
            self.transpose_target()
        self.posteriogram=posteriogram
        self.crop_target(note_range)
        self.even_up_rolls()
        return

    def get_corresp_filename_from_maps(self,filename):
        name=os.path.splitext(filename)[0]
        return name+'_corresp.txt'

    def get_aligned_pianoroll(self,pm_data,corresp,section=None):
        #Makes a quantised piano-roll

        pr = pm_data.get_piano_roll(fs=100,times=corresp[:,0])
        # pr = pm_data.get_piano_roll(fs=100)
        # pr = (pr > 5).astype(int)
        # max_len=int(corresp[-1,0]*100)
        # pr = np.pad(pr,pad_width=((0,0),(0,max_len-pr.shape[1])),mode='constant')
        # pr = self.convert_time_to_note(pr)


        # import display_utils
        # display_utils.compare_piano_rolls([pr],show=True)

        if not section==None:
            #Select the relevant portion of the pianoroll
            begin = section[0]
            end = section[1]
            assert begin < end
            [begin_index, begin_val],[index2, val2] = get_closest(begin,corresp[:,0])
            [end_index, end_val],[index2, val2] = get_closest(end,corresp[:,0])

            pr = pr[:,begin_index:end_index]
            self.begin = begin_val
            self.end = end_val
        else:
            #Do nothing except setting begin and end
            self.begin = 0
            self.end = corresp[-1,0]

        return pr


    def get_aligned_input(self,csv_matrix,corresp,section=None,method='avg'):
        #Makes a quantised input
        #The original input has to be downsampled: the method argument
        #specifies the downsampling method.

        n_notes = csv_matrix.shape[0]
        aligned_input = np.zeros([n_notes,corresp.shape[0]])

        def fill_value(sub_input,i):
            #Computes the value of the note-based input, and puts it in the matrix
            #sub_input is the portion of the input corresponding to the current sixteenth note
            #i is the index of the current sixteenth note


            if method=='avg':
                #Take the mean of the values for the current sixteenth note
                value = np.mean(sub_input,axis=1)
            if method=='step':
                #Take the mean of first quarter of the values for the current sixteenth note
                #Focus on the attacks
                step = int(round(0.25*sub_input.shape[1]))
                value = np.mean(sub_input[:,:step],axis=1)
            if method=='exp':
                #Take the values multiplied by an exponentially-decaying window.
                #Accounts for the exponentially-decaying nature of piano notes
                def exp_window(length,end_value=0.05):
                    a = np.arange(0,length)
                    b = pow(0.1,1.0/length)
                    return np.power(np.full(a.shape,b),a)
                window = exp_window(sub_input.shape[1])
                sub_input_window = sub_input*window[np.newaxis,:]
                value = np.sum(sub_input_window,axis=1)
            if method=='max':
                #If a note is active in the considered sixteenth-note time step,
                #(actually: active more than 5% and more than 3 samples, to account for imprecisions of the alignment)
                #then it is active for the whole time step.
                #Used to convert binary inputs from time-based to note-based time steps.

                value_mean = np.mean(sub_input,axis=1)
                value_sum = np.sum(sub_input,axis=1)
                value = (np.logical_and(value_mean>0.05,value_sum>=3)).astype(int)
            if method=='quant':
                #If a note is active more than half of the sixteenth note time step,
                #it is active for the whole time step.
                #Used to quantise binary inputs (ie align onsets and offsets to the closest sixteenth note)
                value = np.mean(sub_input,axis=1)
                value = (value>0.5).astype(int)


            aligned_input[:,i]=value

        for i in range(aligned_input.shape[1]-1):
            begin = corresp[i,0]
            end = corresp[i+1,0]
            begin_index = int(round(begin*100)) #100 is the sampling frequency of the input (to be abstracted)
            end_index = int(round(end*100))
            sub_input = csv_matrix[:,begin_index:end_index]

            if sub_input.shape[1]==0:
                #Used for debugging
                print "error making align input"
                print begin, end
                print begin_index, end_index
                print sub_input.shape
                print csv_matrix.shape

            fill_value(sub_input,i)

        last_begin = corresp[-1,0]
        last_begin_index = int(round(last_begin*100))
        last_sub_input = csv_matrix[:,last_begin_index:]

        #Prevents some warnings when the corresp file is not perfect
        if not last_sub_input.shape[1]==0:
            fill_value(sub_input,-1)

        if not section==None:
            #Select only the relevant portion of the input
            begin = section[0]
            end = section[1]
            assert begin < end
            [begin_index, begin_val],[index2, val2] = get_closest(begin,corresp[:,0])
            [end_index, end_val],[index2, val2] = get_closest(end,corresp[:,0])

            aligned_input = aligned_input[:,begin_index:end_index]

        return aligned_input


    def copy_section(self,section):
        data = copy.deepcopy(self)
        assert section[0] < section[1]
        [begin_index, begin_val],[index2, val2] = get_closest(section[0],self.corresp[:,0])
        [end_index, end_val],[index2, val2] = get_closest(section[1],self.corresp[:,0])
        data.input = self.input[:,begin_index:end_index]
        data.target = self.target[:,begin_index:end_index]
        data.meter_grid = self.meter_grid[:,begin_index:end_index]
        data.length = end_index-begin_index
        data.begin = section[0]
        data.end = section[1]
        return data



    def import_corresp_file_old(self,filename,get_thresh=True,quantise_steps=0.05):
        #Imports the corresp matrix from the _corresp.txt file
        print "Old Method"
        with open(filename) as file:
            lines = []
            for line in file:
                # The rstrip method gets rid of the "\n" at the end of each line
                lines.append(line.rstrip().split("\t"))
        if get_thresh:
            thresh = lines[-1]
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        lines = np.asarray(lines,dtype=object)

        #Remove the stars, replace with -1
        indexes = lines=='*'
        lines[indexes]='-1'

        index_is_int = [True, False, False, True, True,True, False, False, True, True]
        lines[:,index_is_int]=lines[:,index_is_int].astype(np.int)

        index_is_float = [False, True, False, False, False, False, True,False, False, False]
        lines[:,index_is_float]=lines[:,index_is_float].astype(np.float)

        #Round the steps to the closest quantise_steps value
        quantised_times = np.around(lines[:,6].astype(np.float)/quantise_steps)*quantise_steps
        lines[:,6]=quantised_times

        table_ref = lines[:,[1,6]].astype(np.float)

        #Remove duplicates (chords)
        table_ref = np.unique(table_ref,axis=0)


        #Remove misaligned notes
        mask = np.logical_and(table_ref[:,0]>=0,  table_ref[:,1]>=0)
        table_ref = table_ref[mask,:]




        #In case the first note of the MAPS file is not aligned to 0
        if not table_ref[0,1]==0.0:
            table_ref = np.append([[0,0]],table_ref,axis=0)




        if get_thresh:
            thresh = int(thresh[0])

            return table_ref, thresh
        else:
            return table_ref

    def import_corresp_file(self,filename,get_thresh=True,quantise_steps=0.05):
        print "New Method"
        print filename
        with open(filename) as file:
            lines = []
            for line in file:
                # The rstrip method gets rid of the "\n" at the end of each line
                lines.append(line.rstrip().split("\t"))
        if get_thresh:
            thresh = lines[-1]
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        lines = np.asarray(lines,dtype=object)

        #Remove the stars, replace with -1
        indexes = lines=='*'
        lines[indexes]='-1'

        index_is_int = [True, False, False, True, True,True, False, False, True, True]
        lines[:,index_is_int]=lines[:,index_is_int].astype(np.int)

        index_is_float = [False, True, False, False, False, False, True,False, False, False]
        lines[:,index_is_float]=lines[:,index_is_float].astype(np.float)

        pm_filename = 'data/useless/Piano-midi-all/'+self.set_name_from_maps(filename.replace('_corresp.txt','.mid'))+'.mid'
        pm_data = pm.PrettyMIDI(pm_filename)

        PPQ = float(pm_data.resolution)

        ref_notes = []
        ref_intervals = []
        for instr in pm_data.instruments:
            for note in instr.notes:
                note.start = pm_data.time_to_tick(note.start)/PPQ
                note.end = pm_data.time_to_tick(note.end)/PPQ
                ref_notes += [note.pitch]
                ref_intervals += [[note.start,note.end]]

        ref_notes = np.array(ref_notes)
        ref_intervals = np.array(ref_intervals)


        est_notes = lines[:,8].astype(int)
        est_onset = lines[:,6].astype(float)
        est_intervals = np.zeros([est_onset.shape[0],2])
        est_intervals[:,0] = est_onset
        est_intervals[:,1] = est_onset+1


        import mir_eval
        match = mir_eval.transcription.match_notes(ref_intervals,ref_notes,est_intervals,est_notes,onset_tolerance=0.1,pitch_tolerance=0.01,offset_ratio=None)

        table_ref = lines[:,[1,6]].astype(np.float)

        for i,j in match:
            table_ref[j,1]=ref_intervals[i,0]


        #Round the steps to the closest quantise_steps value
        quantised_times = np.around(table_ref[:,1].astype(np.float)/quantise_steps)*quantise_steps
        table_ref[:,1]=quantised_times

        #Remove duplicates (chords)
        table_ref = np.unique(table_ref,axis=0)


        #Remove misaligned notes
        mask = np.logical_and(table_ref[:,0]>=0,  table_ref[:,1]>=0)
        table_ref = table_ref[mask,:]


        #In case the first note of the MAPS file is not aligned to 0
        if not table_ref[0,1]==0.0:
            table_ref = np.append([[0,0]],table_ref,axis=0)


        if get_thresh:

            thresh = int(float(thresh[0]))

            return table_ref, thresh
        else:
            return table_ref




    def make_corresp_table(self,filename,length_time=None,length_steps=None,quant_step=0.25,annot=None):
        #Makes the corresp table with the right format
        #(ie no repetitions, time steps ordered chronologically, etc)
        #length_time is the length in seconds of the table
        #length_steps is the length in steps of the table
        #length_steps has priority over length_time

        if annot is None:
            corresp_filename = self.get_corresp_filename_from_maps(filename)
            table_ref, thresh = self.import_corresp_file(corresp_filename)

            # #Remove duplicates in the reference list
            # #(sometimes, two close notes get aligned to the same ref point, so it appears twice)
            # steps_ref, indexes_ref = np.unique(table_ref[:,1],return_index=True)
            # steps_ref = list(steps_ref)


            # #Remove deviations (even more robust)
            # #Use a 20 samples window to find closest value
            # #Assume deviations are slowly moving (ie next deviation is going to be within 0.05 of the previous )
            # N = table_ref.shape[0]
            # step = 0
            # dev = 0
            # i = 0
            # table_ref_nodev = np.zeros_like(table_ref)
            # for i in range(N):
            #
            #     (time,step)=table_ref[i]
            #     # import pdb; pdb.set_trace()
            #     table_ref_nodev[i,0] = time
            #
            #     pos_dev = (step-dev)%quant_step
            #     neg_dev = -(step-dev)%quant_step
            #     min_dev = min(pos_dev,neg_dev)
            #     if min_dev==pos_dev:
            #         new_dev = min_dev
            #     else:
            #         new_dev = -min_dev
            #
            #     if abs(new_dev) < 0.001:
            #         #Assume it is deviated, update deviation
            #
            #         table_ref_nodev[i,1]=step-dev
            #     else:
            #         #Assume it is another note, just remove deviation
            #         table_ref_nodev[i,1]=step-dev
            #     print time,step,dev
            #
            # table_ref = table_ref_nodev

            # Only keep the references that are on the grid (gives more robust results)
            a = table_ref*(1/quant_step)
            indexes_ref = np.where(a%1==0)[0]
            # print table_ref
            # print indexes_ref
            steps_ref = list(table_ref[indexes_ref,1])

            # print steps_ref

            # Compute the whole table (adding the ref points not present in the MIDI file)
            steps = list(np.arange(0,table_ref[-1,1]+0.1,quant_step))
            times = []
            ptr=0
            max_ptr = table_ref.shape[0]
            for step in steps:
                #Speed-up : Do not explore the points already seen
                subtable=steps_ref[ptr:]
                if step in subtable:
                    index = subtable.index(step)
                    index_ref = indexes_ref[index+ptr]
                    time = table_ref[index_ref,0]
                    times+=[time]
                    ptr = index #Once the value has been seen, only consider the next values

                else:
                    [index1,ref1], [index2,ref2]=get_closest(step,subtable)
                    index_ref1 = indexes_ref[index1+ptr]
                    index_ref2 = indexes_ref[index2+ptr]
                    time1 = table_ref[index_ref1,0]
                    time2 = table_ref[index_ref2,0]
                    tempo = (time1-time2)/(ref1-ref2)

                    time = min(time1,time2) + (step-min(ref1,ref2))*tempo

                    times+=[time]

            # steps_ref = table_ref[:,1]
            # steps = np.arange(0,table_ref[-1,1]+0.1,quant_step)
            # times = np.zeros_like(steps)
            # # index of steps in the reference that are on the grid
            # index_both = np.where(np.isin(steps_ref,steps))[0]
            #
            # for i, index_ref in enumerate(index_both):
            #     if i<len(index_both)-1:
            #         next_index_ref = index_both[i+1]
            #
            #         if next_index_ref == index_ref+1:
            #             #Just set current index
            #             step = steps_ref[index_ref]
            #             index_step = int(round(step/quant_step))
            #             times[index_step] = table_ref[index_ref,0]
            #         else:
            #
            #             #set current index and interpolate until next ref step
            #             diff_index = (next_index_ref-index_ref)
            #             diff_steps = diff_index*quant_step
            #             diff_time = table_ref[next_index_ref,0] - table_ref[index_ref,0]
            #             #Current tempo value
            #             tempo = diff_time/diff_steps
            #
            #             # print index_ref
            #             # print next_index_ref
            #             # print table_ref[index_ref,0]
            #             # print table_ref[next_index_ref,0]
            #
            #             step = steps_ref[index_ref]
            #             index_step = step/quant_step
            #             time = table_ref[index_ref,0]
            #             times[index_ref:next_index_ref] = (np.arange(0,diff_index)*quant_step)*tempo + time
            #
            #     else:
            #         #Fill rest of array using last tempo value
            #         next_index_ref = steps.shape[0]
            #         step = steps_ref[index_ref]
            #         index_step = step/quant_step
            #         time = table_ref[index_ref,0]
            #         diff_index = (next_index_ref-index_ref)
            #         # print index_ref, next_index_ref
            #
            #         times[index_ref:next_index_ref] = (np.arange(0,diff_index)*quant_step)*tempo + time
            #
            # steps = list(steps)
            # times = list(times)
            #
            # print '___________'
            # print steps
            # print '___________'
            # print times
        else:
            print "Loaded from pickle"
            corresp = annot['corresp_table']
            thresh = annot['threshold']
            times = list(corresp[:,0])
            steps = list(corresp[:,1])

        #Adjust to total_length
        if not length_steps==None:
            if length_steps<len(steps):
                times = times[0:length_steps]
                steps = steps[0:length_steps]
            else:
                to_add = length_steps-len(steps)
                last_timestep = times[-1]-times[-2]
                for i in range(to_add):
                    times += [times[-1]+last_timestep]
                    steps += [steps[-1]+quant_step]
        elif not length_time==None:
            if length_time<times[-1]:
                value = next(x for x in times if x>length_time)
                index = times.index(value)
                times = times[0:index]
                steps = steps[0:index]
            else:
                last_timestep = times[-1]-times[-2]
                while times[-1]+last_timestep<length_time:
                    times += [times[-1]+last_timestep]
                    steps += [steps[-1]+quant_step]

        table = np.transpose(np.asarray([times,steps]),[1,0])
        # print table
        # print thresh
        return table, thresh



    def set_name_from_maps(self,filename):
        name = filename.split('-')[1:]
        name = '-'.join(name)
        name = name.split('_')[:-1]
        name = '_'.join(name)
        self.name =  name
        return name



    def set_sigs_and_keys(self,annotations):
        self.sigs = annotations['time_sig_list']
        self.keys = annotations['key_sig_list']
        return



    def sig_times_to_index(self,key_list,fs):

        sig_times = [time for (sig,time) in key_list]
        sig_times = np.array(sig_times)
        sig_index = np.zeros(sig_times.shape,dtype=int)
        if not self.quant:
            #Time-based time steps
            sig_index = np.round(sig_times*fs).astype(int)
        else:
            corresp = self.corresp
            for i, (key,time) in enumerate(key_list):
                corresp_index = np.argmin(np.abs(corresp[:,0]-time))
                sig_index[i] = np.round(corresp[corresp_index,1]*fs).astype(int)
        return sig_index

    def set_meter_grid(self,fs):
        sig_list = self.sigs
        sig_list_values = [sig for (sig,time) in sig_list]
        sig_list_index = self.sig_times_to_index(sig_list,fs)
        sig_list_len = len(sig_list)
        if self.quant:
            length = self.length
        else:
            length = self.corresp.shape[0]

        meter_grid_quant = np.zeros([4,length])

        for i, (sig,start) in enumerate(zip(sig_list_values,sig_list_index)):
            # import pdb ; pdb.set_trace()
            if start < length:
                if i == sig_list_len-1:
                    end = length
                else:
                    end = min(sig_list_index[i+1],length)

            bar = signature_to_metrical_grid(sig)
            bar_length = bar.shape[1]

            if end == length:
                remain = (end-start)%bar_length
                to_add = bar[:,:remain]
                meter_grid_quant[:,start:end]=np.concatenate([np.tile(bar,[1,(end-start)/bar_length]),to_add],axis=1)
            else:
                try:
                    assert (end-start)%bar_length == 0
                except AssertionError:
                    print "oops"
                    print start, end, end-start
                    print bar_length
                    print sig
                    print (end-start)/bar_length,(end-start)%bar_length
                    raise AssertionError
            # print type(start)
            # print type(end)
                meter_grid_quant[:,start:end]=np.tile(bar,[1,(end-start)/bar_length])

        if not self.quant:
            steps = np.around(self.corresp[:,0]*fs).astype(int)
            meter_grid = np.zeros([4,self.length])
            meter_grid[:,steps] = meter_grid_quant


        if self.quant:
            self.meter_grid = meter_grid_quant
        else:
            self.meter_grid = meter_grid

    def binarize_target(self,thresh=0):
        roll = self.target
        self.target= (roll>thresh).astype(int)
        return

    def even_up_rolls(self):
        #Makes input and target of same size.
        len_input = self.input.shape[1]
        len_target = self.target.shape[1]
        if len_input > len_target:
            self.zero_pad('target',len_input)
            self.length = len_input
        else:
            self.zero_pad('input',len_target)
            self.length = len_target
        return

    def crop_target(self,note_range):
        #Adjusts the range of notes of the target
        if self.note_range != note_range:
            old_note_range = self.note_range
            roll = self.target
            min1 = old_note_range[0]
            max1 = old_note_range[1]
            min2 = note_range[0]
            max2 = note_range[1]

            if min1<min2:
                new_roll = roll[min2-min1:,:]
            else:
                new_roll = np.append(np.zeros([min1-min2,roll.shape[1]]),roll,0)

            if max1<=max2:
                new_roll = np.append(new_roll,np.zeros([max2-max1,roll.shape[1]]),0)
            else:
                new_roll = new_roll[:-(max1-max2),:]

            self.target = new_roll
            self.note_range = note_range
        return

    def zero_pad(self,data,length):
        #Makes the piano-roll of given length
        #Cuts if longer, zero-pads if shorter
        #DO NOT change self.length !!

        roll = getattr(self,data)
        if roll.shape[1] >= length:
            roll_padded = roll[:,0:length]
        else :
            roll_padded = np.pad(roll,pad_width=((0,0),(0,length-roll.shape[1])),mode='constant')
        setattr(self,data,roll_padded)
        return


    def cut(self,len_chunk,keep_padding=True):
        #Returns the roll cut in chunks of len_chunk elements, as well as
        #the list of lengths of the chunks
        #The last element is zero-padded to have len_chunk elements

        roll = self.roll
        if keep_padding:
            size = roll.shape[1]
        else:
            size = self.length
        N_notes = roll.shape[0]
        n_chunks = int(np.ceil(float(size)/len_chunk))

        roll_cut = np.zeros([n_chunks,N_notes,len_chunk])
        lengths = np.zeros([n_chunks])

        j = 0
        n = 0
        length = self.length
        while j < size:
            lengths[n] = min(length,len_chunk)
            length = max(0, length-len_chunk)
            if j + len_chunk < size:
                roll_cut[n]= roll[:,j:j+len_chunk]
                j += len_chunk
                n += 1
            else : #Finishing clause : zero-pad the remaining
                roll_cut[n,:,:]= np.pad(roll[:,j:size],pad_width=((0,0),(0,len_chunk-(size-j))),mode='constant')
                j += len_chunk
        return roll_cut, lengths

    def normalize_input(self,mean,var):
        input_data = self.input
        self.input = (input_data-mean[:,np.newaxis])/var[:,np.newaxis]

    def convert_note_to_time(self,pianoroll,fs,max_len=None):
        #Converts a pianoroll from note-based to time-based time steps,
        #using the corresp table.
        corresp = self.corresp

        #Set length of resulting piano-roll
        if max_len==None:
            length = corresp[-1,0]
            n_steps = corresp.shape[0]
        else:
            length = max_len
            [n_steps,val], _  = get_closest(max_len,list(corresp[:,0]))
        n_notes = pianoroll.shape[0]
        n_times = int(round(length*fs))

        time_roll = np.zeros([n_notes,n_times])

        for i in range(n_steps-1):
            time1, step1 = corresp[i,:]
            time2, step2 = corresp[i+1,:]

            index1 = int(round(time1*fs))
            index2 = int(round(time2*fs))

            active = pianoroll[:,i:i+1] #do this to keep the shape [88,1] instead of [88]
            time_roll[:,index1:index2]=np.repeat(active,index2-index1,axis=1)

        last_time = corresp[n_steps,0]
        last_index = int(round(last_time*fs))
        last_active = np.transpose([pianoroll[:,n_steps-1]],[1,0])

        time_roll[:,last_index:]=np.repeat(last_active,max(n_times-last_index,0),axis=1)

        return time_roll

    def convert_time_to_note(self,pianoroll,max_len=None):
        #Converts a pianoroll from time-based to note-based time steps
        #(ie quantise a piano-roll)
        if max_len is None:
            section = None
        else:
            section = [0,max_len]
        return self.get_aligned_input(pianoroll,self.corresp,section=section,method='quant')



    def transpose_target(self):
        #Transpose a pianoroll (when using data augmentation)
        roll = self.target
        diff = self.transp
        if diff>0:
            roll_transp = np.append(roll[diff:,:],np.zeros([diff,roll.shape[1]]),0)
        elif diff<0:
            roll_transp = np.append(np.zeros([-diff,roll.shape[1]]),roll[:diff,:],0)
        #if diff == 0 : do nothing
        self.target=roll_transp
        return

def special_time(name,time):
    time = round(time,9)
    if 'liz_et1' in name:
        times = {6.794217575: 7,
            265.270355675: 367}
    elif 'liz_et2' in name:
        times= {27.5399385: 30,
            177.186922842: 198}
    elif 'liz_et_trans4' in name:
        times= {306.314562444: 678.5}
    elif 'liz_rhap10' in name:
        times= {12.621430733: 15,
            186.596749625: 262.5}
    elif 'liz_rhap12' in name:
        times= {381.851544325: 532}
    elif 'muss_3' in name:
        times= {24.325015: 38}
    elif 'muss_5' in name:
        times= {7.633271: 11,
            32.0519555: 47}
    elif 'pathetique_1' in name:
        times= {498.061528823: 917}
    elif 'schumm-6' in name:
        times= {95.454291975: 198,
            293.4682612: 609}
    elif 'ty_juni' in name:
        times= {112.190075833: 192}
    elif 'alb_esp3' in name:
        times= {49.090038508: 94.5}
    elif 'alb_se3' in name:
        times= {126.538716133:  225}
    else:
        raise ValueError('Name not in special values!')

    val = times[time]
    return val

def signature_to_metrical_grid(sig):
    num = sig[0]
    denom = sig[1]
    binary_sigs = [(2,4),(3,4),(4,4),(6,4),(8,4)]
    ternary_sigs = [(3,8),(6,8),(9,8),(12,8)]
    if sig in binary_sigs:
        grid = np.zeros([4,num*4])
        j = np.arange(0,grid.shape[1])
        grid[0,:] = 1
        grid[1,j%2==0] = 1
        grid[2,j%4==0] = 1
        grid[3,0] = 1
    elif sig in ternary_sigs:
        grid = np.zeros([4,num*2])
        j = np.arange(0,grid.shape[1])
        grid[0,:] = 1
        grid[1,j%2==0] = 1
        grid[2,j%6==0] = 1
        grid[3,0] = 1

    elif sig == (4,8):
        grid = np.zeros([4,num*2])
        j = np.arange(0,grid.shape[1])
        grid[0,:] = 1
        grid[1,:] = 1
        grid[2,j%2==0] = 1
        grid[3,0] = 1
    else:
        #duration of a bar in 16th notes
        duration = (16.0/denom) * num
        if not duration % 1 ==0:
            #Unusual signature (less than 1% of dataset):
            #just mark the 16th notes and beginning of bar
            duration = 6 #One special case where signature is (11,32)
        else:
            duration = int(duration)

        grid = np.zeros([4,duration])
        grid[0,:] = 1
        grid[:,0] = 1
    return grid

# print signature_to_metrical_grid((5,4))

def get_closest(e,l):
    #Get index of closest element in list l from value e
    #l has to be ordered
    #first output is the closest [index, value]
    #second output is the second closest [index, value]

    if 'numpy.ndarray' in str(type(l)):
        l=list(l)

    default_val = l[-1]
    val2 = next((x for x in l if x>=e),default_val)
    index2 = l.index(val2)
    if index2==0:
        index1 = index2+1
    else:
        index1 = index2-1
    val1 = l[index1]


    if abs(val2-e) < abs(e-val1):
        return [index2, val2], [index1, val1]
    else:
        return [index1, val1],[index2, val2]

def import_csv(filename):
    #Generic function to import a csv file as a numpy matrix
    import csv
    csvfile = open(filename,'rb')
    reader = csv.reader(csvfile,delimiter=",")
    matrix = []
    for row in reader:
        matrix += [row]
    return np.asarray(matrix).astype(float)




#data/Config1/fold1/train/MAPS_MUS-chpn_op66_AkPnBcht.mid
#data/Config1/fold1/train/MAPS_MUS-mz_333_3_ENSTDkCl.mid
#data/Config1/fold1/train/MAPS_MUS-schuim-1_AkPnStgb.mid
#data/Config1/fold1/train/MAPS_MUS-schuim-3_AkPnStgb.mid
#data/Config1/fold1/train/MAPS_MUS-chpn_op25_e2_AkPnBcht.mid
#data/Config1/fold1/train/MAPS_MUS-mz_331_1_SptkBGCl.mid
#data/Config2/fold4/train/MAPS_MUS-appass_1_SptkBGCl.mid
filename = 'data/Config1/fold1/train/MAPS_MUS-chpn_op25_e2_AkPnBcht.mid'


def get_name_from_maps(filename):
    name = filename.split('-')[1:]
    name = '-'.join(name)
    name = name.split('_')[:-1]
    name = '_'.join(name)
    return name


input_folder = ['data/Config1/fold4/train','data/Config1/fold4/test']





def special_time(name,time):
    time = round(time,9)
    if 'liz_et1' in name:
        times = {6.794217575: 6,
            265.270355675: 366}
    elif 'liz_et2' in name:
        times= {27.5399385: 30,
            177.186922842: 198}
    elif 'liz_et_trans4' in name:
        times= {306.314562444: 678.5}
    elif 'liz_rhap10' in name:
        times= {12.621430733: 15,
            186.596749625: 262.5}
    elif 'liz_rhap12' in name:
        times= {381.851544325: 532}
    elif 'muss_3' in name:
        times= {24.325015: 38}
    elif 'muss_5' in name:
        times= {7.633271: 11,
            32.0519555: 47}
    elif 'pathetique_1' in name:
        times= {498.061528823: 917}
    elif 'schumm-6' in name:
        times= {95.454291975: 198,
            293.4682612: 609}
    elif 'ty_juni' in name:
        times= {112.190075833: 192}
    elif 'alb_esp3' in name:
        times= {49.090038508: 94.5}
    elif 'alb_se3' in name:
        times= {126.538716133:  225}
    else:
        raise ValueError('Name not in special values!')

    val = times[time]
    return val

filename = 'data/Config1/fold2/train/MAPS_MUS-liz_et6_StbgTGd2.mid'



# data = DataMaps()
# import cPickle as pickle
# with open('corresp_dataset/full_dataset.p', 'r') as file:
#     annot = pickle.load(file)
# data.make_from_file(filename,4,None,quant=True)
# annot = annot[os.path.basename(filename)]
# print data.length
# print annot['time_sig_list']
# # import display_utils
# # display_utils.compare_piano_rolls([data.target[:,0:50]],show=True)
#
# data.make_from_file(filename,4,None,quant=True,annot=annot)

# for filename in filename_list:

# for subfolder in input_folder:
#     for input_file in os.listdir(subfolder):
#             if input_file.endswith('.mid') and not input_file.startswith('.') and not input_file.endswith('_GT.mid'):
#
#
#                 filename = os.path.join(subfolder, input_file)
#                 # filename = 'data/Config1/fold1/train/MAPS_MUS-chp_op31_AkPnBcht.mid'
#                 # filename_pm = filename
#
#
#                 print filename
#
#
#
#                 # data = DataMaps()
#                 # corresp, _ =corresp, thresh = data.make_corresp_table(filename)
#
#
#
#                 pm_folder = "data/useless/Piano-midi-all"
#                 name_pm = get_name_from_maps(filename)+'.mid'
#                 filename_pm = os.path.join(pm_folder,name_pm)
#                 print filename_pm
#
#                 midi_data = pm.PrettyMIDI(filename_pm)
#
#
#                 keys = midi_data.key_signature_changes
#                 sigs = midi_data.time_signature_changes
#
#                 # table = make_corresp_table(filename_pm.replace('.mid','_corresp.txt'))
#                 table = import_corresp_file(filename_pm.replace('.mid','_corresp.txt'))
#                 # table[:,[2,1]] = table[:,[1,2]]
#
#                 # print table[:10,:]
#                 # print corresp[:10,:]
#
#                 times_k = []
#                 times_s =[]
#
#
#                 for sig in sigs:
#                     # print sig.time
#                     # print table[:,0]
#                     # print (np.round(table[:,0],3)==round(sig.time,3)).shape
#                     # print round(sig.time,3)
#                     # print np.round(table[:,0],3)
#                     if sig.time==0:
#                         time = 0.0
#
#                     else:
#                         mask = np.abs(table[:,0]-sig.time)<0.005
#                         time = table[mask,1]
#                         #remove duplicates when two notes are aligned to the same point
#                         time = np.unique(time)
#
#                         if not len(time) == 1:
#                             print time
#                             try:
#                                 time = [special_time(filename,sig.time)]
#                             except ValueError:
#                                 print "NAAAAAAAAAAAAAAAAAAAA"
#                                 print sig
#                                 print sig.time
#                                 m, s = divmod(sig.time, 60)
#                                 print m,s
#                                 ((idx1,time1),(idx2,time2)) = get_closest(sig.time,table[:,0])
#                                 print table[[idx1,idx2],:]
#                                 # val1 = table[idx1,1]
#                                 # val2 = table[idx2,1]
#                                 # interp_val = ((val2-val1)/(time2-time1))*(sig.time-time1)+val1
#                                 #
#                                 # print "interp_val", interp_val
#
#                                 # tick = midi_data.time_to_tick(sig.time)
#                                 # val = midi_quantised.tick_to_time(tick)
#                                 # print val
#
#
#                     times_s += [time]
#                 for key in keys:
#                     # print sig.time
#                     # print table[:,0]
#                     # print (np.round(table[:,0],3)==round(sig.time,3)).shape
#                     # print round(sig.time,3)
#                     # print np.round(table[:,0],3)
#                     if key.time==0:
#                         time = 0.0
#
#                     else:
#                         mask = np.abs(table[:,0]-key.time)<0.01
#                         time = table[mask,1]
#                         #remove duplicates when two notes are aligned to the same point
#                         time = np.unique(time)
#
#                         if not len(time) == 1:
#                             try:
#                                 time = [special_time(filename,key.time)]
#                             except ValueError:
#                                 print "WAAAAAAAAAAAAAAAAAAAA"
#                                 print key
#                                 print key.time
#                                 m, s = divmod(key.time, 60)
#                                 print m,s
#                                 ((idx1,val1),(idx2,val2)) = get_closest(key.time,table[:,0])
#                                 print table[[idx1,idx2],:]
#
#                     times_k += [time]


# data/useless/Piano-midi-all/liz_et1.mid
# data/useless/Piano-midi-all/liz_et2.mid
# data/useless/Piano-midi-all/liz_et_trans4.mid
# data/useless/Piano-midi-all/liz_rhap10.mid
# data/useless/Piano-midi-all/liz_rhap12.mid
# data/useless/Piano-midi-all/muss_3.mid
# data/useless/Piano-midi-all/muss_5.mid
# data/useless/Piano-midi-all/mz_333_3.mid
# data/useless/Piano-midi-all/pathetique_1.mid
# data/useless/Piano-midi-all/schumm-6.mid
# data/useless/Piano-midi-all/ty_juni.mid
# data/useless/Piano-midi-all/alb_esp3.mid





#
# diff = corresp[1:,:]-corresp[:-1,:]
# # for row in corresp:
# #     if row[0]<0:
# #         print "WAAAAAAAAAAAAAAAAAA"
#
# for i in range(1,len(corresp)-1):
#     time0 = corresp[i-1,0]
#     time1 = corresp[i,0]
#     time2 = corresp[i+1,0]
#
#     step0 = corresp[i-1,1]
#     step1 = corresp[i,1]
#     step2 = corresp[i+1,1]
#     if time1 > time2:
#         print time0, time1, time2
#         print step0, step1, step2
#         print "keeeeeWAAAAAAAAAAAAAAAAAA"
#
# input_data = data.get_aligned_input(csv_matrix,corresp,method='avg',section = None)
# # print input_data[:,-1]
#
# roll = data.get_aligned_pianoroll(pm_data,corresp,section =None)
# print roll.shape
# print corresp.shape

# import matplotlib.pyplot as plt
# plt.imshow(roll)
# plt.show()

# filename = 'data/Config1/fold1/train/MAPS_MUS-chpn_op25_e2_AkPnBcht.mid'
# data = DataMaps()
# data.make_from_file(filename,fs=4,section=[0,30],note_range=[21,109],quant=True,posteriogram=True,method='avg')
# target_time = data.convert_note_to_time(data.target,100,max_len=30)
#
# midi_data = pm.PrettyMIDI(filename)
# roll = midi_data.get_piano_roll()[21:109,0:3000]
# from display_utils import compare_piano_rolls
# import matplotlib.pyplot as plt
# compare_piano_rolls([target_time,roll],[21,109])
# plt.show()
