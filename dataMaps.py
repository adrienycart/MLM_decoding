# -*- coding: utf-8 -*-

import pretty_midi as pm
import numpy as np
import os
import copy
import pickle


pm.pretty_midi.MAX_TICK = 1e10

class DataMaps:
    """Classe representing a couple (posteriogram, ground truth)."""

    def __init__(self):
        self.input_fs = None
        self.input = [] # posteriogram
        self.target = [] # transcription ground-truth
        self.corresp = [] # correspondance table: position in seconds of each timestep
        self.meter_grid = [] # Meter grid
        self.keys = [] # list of couples (key, time), where key is an integer between 0 and 23 and time is in seconds
        self.sigs = [] #list of couples (sig, time), where sig is a list of size 2 and time is in seconds
        self.name = ""
        self.length = 0 # length of the piece (might be different from len(input) because of zero padding)
        self.duration = 0 #duration in seconds
        self.note_range=[0,128]
        self.begin=0 # beginning of the considered section in seconds
        self.end=0 # end of the considered section in seconds
        self.timestep_type = None
        self.acoutic_model = ""
        self.with_onsets = False

    def make_from_file(self,filename,timestep_type,section=None,method='avg',si_target=False,acoustic_model='benetos',with_onsets=False):
        self.acoustic_model = acoustic_model
        self.with_onsets=with_onsets
        if acoustic_model == 'benetos':
            self.input_fs = 100
            note_range = [21,109]
        elif acoustic_model == 'kelz':
            if with_onsets:
                self.input_fs = 50
            else:
                self.input_fs = 25
            note_range = [21,109]
        elif acoustic_model == 'bittner':
            self.input_fs = 22050.0/256.0
            # note_range = [24,97]
            note_range = [21,109]
        self.timestep_type = timestep_type
        self.set_name_from_maps(filename)

        pm_data = pm.PrettyMIDI(filename)
        input_matrix = self.get_input_matrix(filename)


        self.set_corresp(pm_data,timestep_type)
        self.target = self.get_roll_from_times(pm_data,section)
        self.input = align_matrix(input_matrix,self.corresp,self.input_fs,section,method,self.with_onsets)

        self.set_sigs_and_keys(pm_data)
        self.crop_target(note_range)
        self.even_up_rolls()

        return


    def get_input_matrix(self,filename):
        #Method to get a pretty_midi object and an input matrix from a MIDI filename.
        with_onsets=self.with_onsets
        if self.acoustic_model == 'benetos':
            csv_filename = filename.replace('.mid','_pianoroll.csv')
            input_matrix = np.transpose(np.loadtxt(csv_filename,delimiter=','),[1,0])
        elif self.acoustic_model == 'kelz':
            if with_onsets:
                active_filename = filename.replace('.mid','_active.csv')
                onset_filename = filename.replace('.mid','_onset.csv')
                active_matrix = np.transpose(np.loadtxt(active_filename),[1,0])
                onset_matrix = np.transpose(np.loadtxt(onset_filename),[1,0])
                input_matrix = np.concatenate([active_matrix[:,:,None],onset_matrix[:,:,None]],axis=2)
            else:
                csv_filename = filename.replace('.mid','.csv')
                input_matrix = np.transpose(np.loadtxt(csv_filename),[1,0])
        elif self.acoustic_model == 'bittner':
            #Load matrix
            npz_filename = filename.replace('.mid','_multif0_salience.npz')
            matrix = np.load(npz_filename)['salience']

            # Group frequencies into MIDI pitches
            N = matrix.shape[1]
            n_freqs = matrix.shape[0]

            freqs =  32.7*np.power(2,np.arange(0,n_freqs)/60.0)
            mid_numbers = pm.hz_to_note_number(freqs)
            mid_numbers = np.round(mid_numbers).astype(int)

            max_mid = max(mid_numbers)
            min_mid = min(mid_numbers)

            input_matrix = np.zeros([max_mid+1-min_mid,N])
            for i in range(min_mid,max_mid+1):
                input_matrix[i-min_mid,:] = np.mean(matrix[mid_numbers==i],axis=0)

            # Zero-pad to range [21,109]
            input_note_range = [24,97]
            output_note_range = [21,109]
            input_matrix = np.pad(input_matrix,((24-21,109-97),(0,0)),'constant')



        return input_matrix



    def get_corresp_filename_from_maps(self,filename):
        name=os.path.splitext(filename)[0]
        return name+'_corresp.txt'

    def get_roll_from_times(self,pm_data,section=None):
        #Makes a quantised piano-roll
        with_onsets=self.with_onsets

        corresp = self.corresp

        roll = np.zeros([128,len(corresp)])

        for instr in pm_data.instruments:
            for note in instr.notes:
                start = np.argmin(np.abs(corresp-note.start))
                end = np.argmin(np.abs(corresp-note.end))
                if start == end:
                    end = start+1
                roll[note.pitch,start:end]=1

                if with_onsets:
                    roll[note.pitch,start] = 2

        if not section==None:
            #Select the relevant portion of the pianoroll
            begin = section[0]
            end = section[1]
            assert begin < end
            [begin_index, begin_val],[index2, val2] = get_closest(begin,corresp)
            [end_index, end_val],[index2, val2] = get_closest(end,corresp)

            roll = roll[:,begin_index:end_index]
            self.begin = begin_val
            self.end = end_val
        else:
            #Do nothing except setting begin and end
            self.begin = 0
            self.end = corresp[-1]

        return roll



    def set_corresp(self,pm_data,timestep_type):
        #Set the correspondance table from pm_data (should be a properly annotated MIDI file)

        end_time = pm_data.get_end_time()
        end_tick = pm_data.time_to_tick(end_time)

        if timestep_type == "quant":
            PPQ = float(pm_data.resolution)
            end_note = end_tick/PPQ
            note_steps = np.arange(0,end_note,0.25)
            tick_steps = np.round(note_steps*PPQ).astype(int)
            corresp = np.zeros_like(tick_steps,dtype=float)
            for i,tick in enumerate(tick_steps):
                corresp[i]=pm_data.tick_to_time(int(tick))
        elif timestep_type == "quant_short":
            PPQ = float(pm_data.resolution)
            end_note = end_tick/PPQ
            note_steps = np.arange(0,end_note,1.0/12)
            tick_steps = np.round(note_steps*PPQ).astype(int)
            corresp = np.zeros_like(tick_steps,dtype=float)
            for i,tick in enumerate(tick_steps):
                corresp[i]=pm_data.tick_to_time(int(tick))
        elif timestep_type == 'event':
            corresp = np.unique(pm_data.get_onsets())
            #Remove onsets that are within 40ms of each other (keep first one only)
            diff = corresp[1:] - corresp[:-1]
            close = diff<0.04
            while np.any(close):
                to_keep = np.where(np.logical_not(close))
                corresp = corresp[to_keep[0]+1]
                diff = corresp[1:] - corresp[:-1]
                close = diff<0.04
        elif timestep_type == "time":
            fs=25
            corresp = np.arange(0,end_time,1.0/fs)
        elif timestep_type == "20ms":
            fs=50
            corresp = np.arange(0,end_time,1.0/fs)
        else:
            raise  ValueError('Timestep type not understood: '+str(timestep_type))

        self.corresp = corresp
        return


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
        data.duration = section[1]-section[0]
        return data



    def set_name_from_maps(self,filename):
        name = filename.split('-')[1:]
        name = '-'.join(name)
        name = name.split('_')[:-1]
        name = '_'.join(name)
        self.name =  name
        return name



    def set_sigs_and_keys(self,pm_data):
        time_sig_list = pm_data.time_signature_changes
        key_sig_list = pm_data.key_signature_changes

        self.sigs = [((sig.numerator,sig.denominator),sig.time) for sig in time_sig_list]
        self.keys = [(sig.key_number,sig.time) for sig in key_sig_list]
        return


    def sig_times_to_index(self,key_list):
        #NOT CURRENTLY WORKING, NOT USED

        sig_times = [time for (sig,time) in key_list]
        sig_times = np.array(sig_times)
        sig_index = np.zeros(sig_times.shape,dtype=int)
        if self.timestep_type == 'time':
            #Time-based time steps
            sig_index = np.round(sig_times*25).astype(int)
        else:
            corresp = self.corresp
            for i, (key,time) in enumerate(key_list):
                sig_index[i] = np.argmin(np.abs(corresp-time))
        return sig_index

    def set_meter_grid(self):
        #NOT CURRENTLY WORKING, NOT USED
        sig_list = self.sigs
        sig_list_values = [sig for (sig,time) in sig_list]
        sig_list_index = self.sig_times_to_index(sig_list)
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
                    print("oops")
                    print(start, end, end-start)
                    print(bar_length)
                    print(sig)
                    print((end-start)/bar_length,(end-start)%bar_length)
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


    def even_up_rolls(self):
        #Makes input and target of same size.
        len_input = self.input.shape[1]
        len_target = self.target.shape[1]
        if len_input > len_target:
            self.input = self.input[:,:len_target]
            self.length = len_target
        else:
            self.target = self.target[:,:len_input]
            self.length = len_input

        self.duration = self.corresp[-1]
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

    def transpose(self,diff):
        #Returns a copy of self, transposed of diff semitones
        #diff can be positive or negative
        data_trans = copy.deepcopy(self)
        input_data = self.input
        target_data = self.target
        if diff<0:
            data_trans.input = np.append(input_data[-diff:,:],np.zeros([-diff,input_data.shape[1]]),0)
            data_trans.target = np.append(target_data[-diff:,:],np.zeros([-diff,target_data.shape[1]]),0)
        elif diff>0:
            data_trans.input = np.append(np.zeros([diff,input_data.shape[1]]),input_data[:-diff,:],0)
            data_trans.target = np.append(np.zeros([diff,target_data.shape[1]]),target_data[:-diff,:],0)
        #if diff == 0 : do nothing
        return data_trans

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

    def cut(self,len_chunk,keep_padding=True,as_list=False):
        #Returns the roll cut in chunks of len_chunk elements, as well as
        #the list of lengths of the chunks
        #The last element is zero-padded to have len_chunk elements


        if keep_padding:
            size = self.input.shape[1]
        else:
            size = self.length

        if as_list:
            input_cut = []
            target_cut = []
            lengths = []
        else:
            n_chunks = int(np.ceil(float(size)/len_chunk))
            input_cut = np.zeros([n_chunks,self.input.shape[0],len_chunk])
            target_cut = np.zeros([n_chunks,self.input.shape[0],len_chunk])
            lengths = np.zeros([n_chunks])

        j = 0
        n = 0
        length = self.length
        while j < size:
            if as_list:
                lengths += [min(length,len_chunk)]
            else:
                lengths[n] = min(length,len_chunk)
            length = max(0, length-len_chunk)
            if j + len_chunk < size:
                if as_list:
                    input_cut += [self.input[:,j:j+len_chunk]]
                    target_cut += [self.target[:,j:j+len_chunk]]
                else:
                    input_cut[n]= self.input[:,j:j+len_chunk]
                    target_cut[n] = self.target[:,j:j+len_chunk]
                j += len_chunk
                n += 1
            else : #Finishing clause : zero-pad the remaining
                if as_list:
                    input_cut += [np.pad(self.input[:,j:size],pad_width=((0,0),(0,len_chunk-(size-j))),mode='constant')]
                    target_cut += [np.pad(self.target[:,j:size],pad_width=((0,0),(0,len_chunk-(size-j))),mode='constant')]
                else:
                    input_cut[n,:,:]= np.pad(self.input[:,j:size],pad_width=((0,0),(0,len_chunk-(size-j))),mode='constant')
                    target_cut[n,:,:]= np.pad(self.target[:,j:size],pad_width=((0,0),(0,len_chunk-(size-j))),mode='constant')
                j += len_chunk
        return input_cut, target_cut, lengths


    def normalize_input(self,mean,var):
        input_data = self.input
        self.input = (input_data-mean[:,np.newaxis])/var[:,np.newaxis]

    def get_key_profile(self):

        roll = self.target
        shape = roll.shape

        length = roll.shape[1]

        key_profile = np.sum(self.target,axis=1)/float(length)
        # key_profile_matrix = np.tile(key_profile,(length,1)).transpose()


        return key_profile

    def get_key_profile_octave(self):

        roll = self.target
        i = np.arange(roll.shape[0])
        output = np.zeros([12],dtype=float)
        for p in range(12):
            active_pitch_class = np.max(roll[i%12==p,:],axis=0)
            output[p] = np.mean(active_pitch_class)

        return output

class DataMapsBeats(DataMaps):
    def make_from_file(self,filename,gt_beats=False,beat_subdiv=[0.0,1.0/4,1.0/3,1.0/2,2.0/3,3.0/4],section=None,method='avg',si_target=False,acoustic_model='benetos',with_onsets=False):

        self.with_onsets=with_onsets

        if type(beat_subdiv) is str:
            beat_subdiv_str = beat_subdiv
            beat_subdiv_str=beat_subdiv_str.split(',')
            beat_subdiv = []
            for beat_str in beat_subdiv_str:
                if '/' in beat_str:
                    beat_str_split = beat_str.split('/')
                    beat_subdiv += [float(beat_str_split[0])/float(beat_str_split[1])]
                else:
                    beat_subdiv += [float(beat_str)]
        else:
            pass


        self.acoustic_model = acoustic_model
        if acoustic_model == 'benetos':
            self.input_fs = 100
            note_range = [21,109]
        elif acoustic_model == 'kelz':
            if self.with_onsets:
                self.input_fs = 50
            else:
                self.input_fs = 25
            note_range = [21,109]
        elif acoustic_model == 'bittner':
            self.input_fs = 22050.0/256.0
            # note_range = [24,97]
            note_range = [21,109]

        self.set_name_from_maps(filename)

        pm_data = pm.PrettyMIDI(filename)
        input_matrix = self.get_input_matrix(filename)
        beats_filename = filename.replace('.mid','_b_gt.csv') if gt_beats else filename.replace('.mid','_b_est.csv')
        beats = np.loadtxt(beats_filename)

        self.set_corresp(pm_data,beats,beat_subdiv)
        self.target = self.get_roll_from_times(pm_data,section)
        self.input = align_matrix(input_matrix,self.corresp,self.input_fs,section,method,self.with_onsets)


        self.set_sigs_and_keys(pm_data)
        self.crop_target(note_range)
        self.even_up_rolls()

        return

    def set_corresp(self,pm_data,beats,beat_subdiv):
        #Set the correspondance table from pm_data (should be a properly annotated MIDI file)

        # Check that beat_subdiv is correct
        beat_subdiv = sorted(beat_subdiv)

        for i,val in enumerate(beat_subdiv):
            if type(val) is not float:
                raise ValueError('All the beat_subdiv values should be floats!')

        beat_subdiv = np.array(beat_subdiv)
        if beat_subdiv[0] != 0:
            raise ValueError('beat_subdiv[0] should be 0.0!')
        if np.any(np.logical_or(beat_subdiv<0,beat_subdiv>=1)):
            raise ValueError('All beat_subdiv values should be between 0 and 1 (excluded)!')
        if np.any(beat_subdiv[1:]-beat_subdiv[:-1]==0):
            raise ValueError('All beat_subdiv values should be different!')

        end_time = pm_data.get_end_time()
        end_tick = pm_data.time_to_tick(end_time)

        # Make step times from beats and beat_subdiv
        n_subdiv=len(beat_subdiv)
        n_beats = len(beats)-1 #Only take beat intervals that have an end

        beat_duration = beats[1:]-beats[:-1]
        offset_to_beat = np.tile(beat_subdiv,n_beats)*np.repeat(beat_duration,n_subdiv)
        corresp = np.repeat(beats[:-1],n_subdiv) + offset_to_beat

        self.corresp = corresp
        return

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


def align_matrix(input_matrix,corresp,input_fs,section=None,method='avg',with_onsets=False):
    #Makes a quantised input
    #The original input has to be downsampled: the method argument
    #specifies the downsampling method.


    n_notes = input_matrix.shape[0]
    end_sec = min(input_matrix.shape[1]/float(input_fs),corresp[-1])
    (n_steps,_),_ = get_closest(end_sec,corresp)

    if with_onsets:
        aligned_input = np.zeros([n_notes,n_steps,2])
    else:
        aligned_input = np.zeros([n_notes,n_steps])



    def get_fill_value(sub_input,method):
        #Computes the value of the note-based input, and puts it in the matrix
        #sub_input is the portion of the input corresponding to the current sixteenth note
        #i is the index of the current sixteenth note


        if method=='avg':
            #Take the mean of the values for the current sixteenth note
            value = np.mean(sub_input,axis=1)
        elif method=='step':
            #Take the mean of first quarter of the values for the current sixteenth note
            #Focus on the attacks
            step = max(int(round(0.25*sub_input.shape[1])),1)
            value = np.mean(sub_input[:,:step],axis=1)
        elif method=='exp':
            #Take the values multiplied by an exponentially-decaying window.
            #Accounts for the exponentially-decaying nature of piano notes
            def exp_window(length,end_value=0.05):
                a = np.arange(0,length)
                b = pow(0.1,1.0/length)
                return np.power(np.full(a.shape,b),a)
            window = exp_window(sub_input.shape[1])
            sub_input_window = sub_input*window[np.newaxis,:]
            value = np.sum(sub_input_window,axis=1)
        elif method=='max':
            #If a note is active in the considered sixteenth-note time step,
            #(actually: active more than 5% and more than 3 samples, to account for imprecisions of the alignment)
            #then it is active for the whole time step.
            #Used to convert binary inputs from time-based to note-based time steps.

            # value_mean = np.mean(sub_input,axis=1)
            # value_sum = np.sum(sub_input,axis=1)
            # value = (np.logical_and(value_mean>0.05,value_sum>=3)).astype(int)
            value = np.max(sub_input,axis=1)
        elif method=='quant':
            #If a note is active more than half of the sixteenth note time step,
            #it is active for the whole time step.
            #Used to quantise binary inputs (ie align onsets and offsets to the closest sixteenth note)
            value = np.mean(sub_input,axis=1)
            value = (value>0.5).astype(int)
        return value

    for i in range(aligned_input.shape[1]-1):
        begin = corresp[i]
        end = corresp[i+1]
        begin_index = int(round(begin*input_fs)) #input_fs is the sampling frequency of the input
        end_index = max(int(round(end*input_fs)),begin_index+1) #We want to select at least 1 frame of the input
        # if with_onsets:
        #     sub_input = input_matrix[:,begin_index:end_index,:]
        # else:
        sub_input = input_matrix[:,begin_index:end_index]

        if sub_input.shape[1]==0:
            #Used for debugging
            print("error making align input")
            print(begin, end,end-begin)
            print(begin_index, end_index)
            print(begin*input_fs,end*input_fs)
            print(sub_input.shape)
            print(input_matrix.shape)

        if with_onsets:
            aligned_input[:,i,0] = get_fill_value(sub_input[:,:,0],method)
            # The onsets is taken as the max of a range centered on corresp[i]
            # This boils down to hard-quantising the onsets to the closest subdivision
            if i==0:
                begin_onset = (corresp[i])/2
                end_onset  = (corresp[i+1]+corresp[i])/2
            else:
                begin_onset  = (corresp[i]+corresp[i-1])/2
                end_onset  = (corresp[i+1]+corresp[i])/2
            begin_index_onset = int(round(begin_onset *input_fs)) #input_fs is the sampling frequency of the input
            end_index_onset = max(int(round(end_onset *input_fs)),begin_index_onset+1) #We want to select at least 1 frame of the input
            sub_input_onset = input_matrix[:,begin_index_onset:end_index_onset,1]


            aligned_input[:,i,1] = get_fill_value(sub_input_onset,'max')
        else:
            aligned_input[:,i] = get_fill_value(sub_input,method)

    last_begin = corresp[-1]
    last_begin_index = int(round(last_begin*input_fs))
    last_sub_input = input_matrix[:,last_begin_index:]

    #Prevents some warnings when the corresp file is not perfect
    if not last_sub_input.shape[1]==0:
        if with_onsets:
            aligned_input[:,i,0] = get_fill_value(sub_input[:,:,0],method)
            aligned_input[:,i,1] = get_fill_value(sub_input[:,:,1],'max')
        else:
            aligned_input[:,-1] = get_fill_value(sub_input,method)

    if not section==None:
        #Select only the relevant portion of the input
        begin = section[0]
        end = section[1]
        assert begin < end
        [begin_index, begin_val],[index2, val2] = get_closest(begin,corresp)
        [end_index, end_val],[index2, val2] = get_closest(end,corresp)

        aligned_input = aligned_input[:,begin_index:end_index]

    return aligned_input

def convert_note_to_time(pianoroll,corresp,input_fs,max_len=None):
    #Converts a pianoroll from note-based to time-based time steps,
    #using the corresp table.

    fs=input_fs

    #Set length of resulting piano-roll
    if max_len==None:
        length = corresp[-1]
        n_steps = corresp.shape[0]
    else:
        length = min(max_len, corresp[-1])
        [n_steps,val], _  = get_closest(max_len,list(corresp))
    n_notes = pianoroll.shape[0]
    n_times = int(round(length*fs))

    time_roll = np.zeros([n_notes,n_times])

    for i in range(n_steps-1):
        time1 = corresp[i]
        time2 = corresp[i+1]

        index1 = int(round(time1*fs))
        index2 = int(round(time2*fs))

        active = pianoroll[:,i:i+1] #do this to keep the shape [88,1] instead of [88]
        time_roll[:,index1:index2]=np.repeat(active,index2-index1,axis=1)

    last_time = corresp[n_steps]
    last_index = int(round(last_time*fs))
    last_active = np.transpose([pianoroll[:,n_steps-1]],[1,0])

    time_roll[:,last_index:]=np.repeat(last_active,max(n_times-last_index,0),axis=1)

    return time_roll


def get_name_from_maps(filename):
    name = filename.split('-')[1:]
    name = '-'.join(name)
    name = name.split('_')[:-1]
    name = '_'.join(name)
    return name


# filename = 'data/outputs_adsr_split20p/lr_0.15_bs_25665068/test/MAPS_MUS-alb_se2_ENSTDkCl.mid'
# np.seterr(all='raise')
# # data = DataMaps()
# # data.make_from_file(filename,'time',[0,10],with_onsets=True,acoustic_model='kelz')
# # print data.input.shape

# data = DataMapsBeats()
# data.make_from_file(filename,section=[0,10],with_onsets=True,acoustic_model='kelz')
# #print data.input.shape

# onset_filename = filename.replace('.mid','_onset.csv')
# onset_matrix = np.transpose(np.loadtxt(onset_filename),[1,0])

# corresp_steps= np.round(data.corresp*50)
# corresp_half_steps = np.round((data.corresp[1:]+data.corresp[:-1])*50/2)

# import matplotlib.pyplot as plt
# fig, [ax0,ax1,ax2,ax3] = plt.subplots(4,1)
# ax0.imshow(onset_matrix[:,:10*50],aspect='auto',origin='lower')
# for i in corresp_steps[:60]:
#     ax0.plot([i,i],[0,87],color='black',linewidth=0.5)
# for i in corresp_half_steps[:60]:
#     ax0.plot([i,i],[0,87],color='grey',linewidth=0.5)
# ax1.imshow(data.input[:,:,0],aspect='auto',origin='lower')
# ax2.imshow(data.input[:,:,1],aspect='auto',origin='lower')
# ax3.imshow(data.target,aspect='auto',origin='lower')
# plt.show()

# data = DataMapsBeats()
# data.make_from_file(filename,section=[0,10],with_onsets=True,acoustic_model='kelz')
# #print data.input.shape

# import matplotlib.pyplot as plt
# fig, [ax1,ax2,ax3] = plt.subplots(3,1)
# ax1.imshow(data.input[:,:,0],aspect='auto',origin='lower')
# ax2.imshow(data.input[:,:,1],aspect='auto',origin='lower')
# ax3.imshow(data.target,aspect='auto',origin='lower')
# plt.show()



# import matplotlib.pyplot as plt
# fig, [ax1,ax2] = plt.subplots(2,1)
# ax1.imshow(data.input,aspect='auto',origin='lower')
# ax2.imshow(data.target,aspect='auto',origin='lower')
# plt.show()


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
# roll = data.get_roll_from_times(pm_data,corresp,section =None)
# print roll.shape
# print corresp.shape

# import matplotlib.pyplot as plt
# plt.imshow(roll)
# plt.show()

# filename = 'data/Config1/fold1/train/MAPS_MUS-chpn_op25_e2_AkPnBcht.mid'
# data = DataMaps()
# data.make_from_file(filename,fs=4,section=[0,30],note_range=[21,109],quant=True,posteriogram=True,method='avg')
# target_time = data.convert_note_to_time(data.target,25,max_len=30)
#
# midi_data = pm.PrettyMIDI(filename)
# roll = midi_data.get_piano_roll()[21:109,0:3000]
# from display_utils import compare_piano_rolls
# import matplotlib.pyplot as plt
# compare_piano_rolls([target_time,roll],[21,109])
# plt.show()
