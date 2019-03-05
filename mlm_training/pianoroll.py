# -*- coding: utf-8 -*-

import pretty_midi as pm
import numpy as np
import os
import copy

class Pianoroll:
    """Classe representing a piano-roll."""

    def __init__(self):
        self.roll = []
        self.name = ""
        self.length = 0
        self.note_range=[0,128]
        self.timestep_type = None
        self.key = 0
        self.key_list = []

    def make_from_file(self,filename,timestep_type,section=None,note_range=[0,128],key_method='main'):
        midi_data = pm.PrettyMIDI(filename)
        self.make_from_pm(midi_data,timestep_type,section,note_range,key_method)
        self.name = os.path.splitext(os.path.basename(filename))[0]
        return

    def make_from_pm(self,data,timestep_type,section=None,note_range=[0,128],key_method='main'):
        self.timestep_type = timestep_type

        #Get the roll matrix
        if timestep_type=="quant":
            self.roll = get_quant_piano_roll(data,4,section)
        elif timestep_type=="event":
            self.roll = get_event_roll(data,section)
        elif timestep_type=="time":
            piano_roll = data.get_piano_roll(25)
            if not section == None :
                min_time_step = int(round(section[0]*25))
                max_time_step = int(round(section[1]*25))
                self.roll = piano_roll[:,min_time_step:max_time_step]
            else:
                self.roll = piano_roll

        self.length = self.roll.shape[1]-1
        self.crop(note_range)
        self.binarize()

        return


    def binarize(self):
        roll = self.roll
        self.roll = np.not_equal(roll,np.zeros(roll.shape)).astype(int)
        return

    def crop(self,note_range):
        if self.note_range != note_range:
            old_note_range = self.note_range
            roll = self.roll
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

            self.roll = new_roll
            self.note_range = note_range
        return

    def zero_pad(self,length):
        #Makes the piano-roll of given length
        #Cuts if longer, zero-pads if shorter
        #DO NOT change self.length !!

        roll = self.roll
        if self.length >= length:
            roll_padded = roll[:,0:length]
        else :
            roll_padded = np.pad(roll,pad_width=((0,0),(0,length-roll.shape[1])),mode='constant')
        self.roll = roll_padded
        return


    def cut(self,roll,len_chunk,keep_padding=True,as_list=False):
        #Returns the roll cut in chunks of len_chunk elements, as well as
        #the list of lengths of the chunks
        #The last element is zero-padded to have len_chunk elements


        if keep_padding:
            size = roll.shape[1]
        else:
            size = self.length

        if as_list:
            roll_cut = []
            lengths = []
        else:
            n_chunks = int(np.ceil(float(size)/len_chunk))
            roll_cut = np.zeros([n_chunks,roll.shape[0],len_chunk])
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
                    roll_cut += [roll[:,j:j+len_chunk]]
                else:
                    roll_cut[n]= roll[:,j:j+len_chunk]
                j += len_chunk
                n += 1
            else : #Finishing clause : zero-pad the remaining
                if as_list:
                    roll_cut += [np.pad(roll[:,j:size],pad_width=((0,0),(0,len_chunk-(size-j))),mode='constant')]
                else:
                    roll_cut[n,:,:]= np.pad(roll[:,j:size],pad_width=((0,0),(0,len_chunk-(size-j))),mode='constant')
                j += len_chunk
        return roll_cut, lengths

    def transpose(self,diff):
        #Returns a copy of self, transposed of diff semitones
        #diff can be positive or negative
        pr_trans = copy.deepcopy(self)
        roll = self.roll
        if diff<0:
            pr_trans.roll = np.append(roll[-diff:,:],np.zeros([-diff,roll.shape[1]]),0)
        elif diff>0:
            pr_trans.roll = np.append(np.zeros([diff,roll.shape[1]]),roll[:-diff,:],0)
        #if diff == 0 : do nothing

        pr_trans.key = (self.key+diff)%12
        pr_trans.key_list = [((key+diff)%12,time) for (key,time) in self.key_list]
        return pr_trans


    def timestretch(self):
        pr_stretch = copy.deepcopy(self)
        roll = self.roll
        length = roll.shape[1]
        #duplicate each column by multiplying by a clever matrix
        a = np.zeros([length,2*length])
        i,j = np.indices(a.shape)
        a[i==j//2]=1
        pr_stretch.roll = np.matmul(roll,a)
        pr_stretch.length = 2*self.length
        return pr_stretch

    def split_pitchwise(self,window):
        seqs = []
        targets = []
        n_notes = self.note_range[1]-self.note_range[0]
        roll = self.roll
        roll_pad = np.pad(roll,((window,window),(0,0)),'constant')
        for i in range(window,window+n_notes):
            seq = np.zeros([2*window+1,self.roll.shape[1]-1])
            seq[:,:] = roll_pad[i-window:i+window+1,:-1]
            seqs += [seq]
            targets += [roll_pad[i:i+1,1:]]
        return seqs,targets


    def get_roll(self):
        return self.roll, self.length

    def get_gt(self):
        return self.roll[:,1:]


def get_quant_piano_roll(midi_data,fs=4,section=None):
    data = copy.deepcopy(midi_data)

    PPQ = float(data.resolution)

    for instr in data.instruments:
        for note in instr.notes:
            note.start = data.time_to_tick(note.start)/PPQ
            note.end = data.time_to_tick(note.end)/PPQ


    # quant_piano_roll = data.get_piano_roll(fs)

    # PROPER WAY OF IMPORTING PIANO-ROLLS !
    length = data.get_piano_roll().shape[1]/100.0
    quant_piano_roll = data.get_piano_roll(times=np.arange(0,length,1/float(fs)))
    quant_piano_roll = (quant_piano_roll>=7).astype(int)


    if not section == None:
        begin = section[0]
        end = section[1]
        assert begin < end

        begin_index = int(round(midi_data.time_to_tick(begin)/PPQ*fs))
        end_index = int(round(midi_data.time_to_tick(end)/PPQ*fs))
        quant_piano_roll = quant_piano_roll[:,begin_index:end_index]

    return quant_piano_roll

def get_event_roll(midi_data,section=None):
    steps = np.unique(midi_data.get_onsets())
    #Round to closest 0.04 value
    steps_round = np.around(steps.astype(np.float)*25)/25
    #Remove duplicates
    _,indexes = np.unique(steps_round,return_index=True)
    steps = steps[indexes]

    pr = midi_data.get_piano_roll(times=steps)
    pr = (pr>=7).astype(int)

    if not section is None:
        #Select the relevant portion of the pianoroll
        begin = section[0]
        end = section[1]
        assert begin < end
        begin_index = np.argmin(np.abs(begin-steps))
        end_index = np.argmin(np.abs(end-steps))
        pr = pr[:,begin_index:end_index]

    return pr





def scale_template(scale,note_range=[21,109]):
    #Returns a 88*1 matrix with True if the corresponding pitch is in the given scale
    #If scale is minor, natural, melodic and harmonic minor are accepted.

    scale = int(scale)
    note_min, note_max = note_range
    key = scale%12
    is_major = scale//12==0
    #Treat everything as in C
    note_min_t = note_min-key
    note_max_t = note_max-key
    octave_max = note_max_t//12
    octave_min = note_min_t//12

    if is_major:
        scale = [0,2,4,5,7,9,11]
    else:
        scale = [0,2,3,5,7,8,9,10,11]
    current_scale = [x+key for x in scale]
    single_notes = []

    for i in range(octave_min,octave_max+1):
        to_add =  [12*i+x for x in scale if 12*i+x>= note_min_t and 12*i+x< note_max_t]
        single_notes = single_notes + to_add
    #Transpose back to the correct key
    output = [x + key for x in single_notes]
    return output






# pr = Pianoroll()
# pr.roll = np.array([[1,1,2,2,3,3],[4,4,5,5,6,6],[1,1,2,2,3,3]])
# pr.length = 6
# pr.note_range=[3,6]
# print pr.split_pitchwise(1)[0]
