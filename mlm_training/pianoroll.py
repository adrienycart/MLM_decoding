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
        self.end_time = 0
        self.note_range=[0,128]
        self.timestep_type = None
        self.key = 0
        self.key_list_times = []
        self.key_list = []
        self.key_profiles_list = []

    def make_from_file(self,filename,timestep_type,section=None,note_range=[0,128],key_method='main',with_onsets=False):
        midi_data = pm.PrettyMIDI(filename)
        self.make_from_pm(midi_data,timestep_type,section,note_range,key_method,with_onsets)
        self.name = os.path.splitext(os.path.basename(filename))[0]
        return

    def make_from_pm(self,data,timestep_type,section=None,note_range=[0,128],key_method='main',with_onsets=False):


        self.timestep_type = timestep_type

        total_duration = data.get_piano_roll().shape[1]/100.0
        end_time = min(section[1],total_duration) if section is not None else total_duration
        self.end_time = end_time

        times = None

        #Get the roll matrix
        if timestep_type=="quant" or "quant_short":
            if timestep_type == "quant":
                fs=4
            elif timestep_type == "quant_short":
                fs=12

            end_tick = data.time_to_tick(end_time)
            PPQ = float(data.resolution)
            end_note = end_tick/PPQ
            note_steps = np.arange(0,end_note,1.0/fs)
            tick_steps = np.round(note_steps*PPQ).astype(int)
            times = np.zeros_like(tick_steps,dtype=float)
            for i,tick in enumerate(tick_steps):
                times[i]=data.tick_to_time(int(tick))
        elif timestep_type=="event":
            times = np.unique(data.get_onsets())
            #Remove onsets that are within 50ms of each other (keep first one only)
            diff = times[1:] - times[:-1]
            close = diff<0.05
            while np.any(close):
                to_keep = np.where(np.logical_not(close))
                times = times[to_keep[0]+1]
                diff = times[1:] - times[:-1]
                close = diff<0.05
        elif timestep_type=="time":
            fs=25
            times = np.arange(0,end_time,1.0/fs)

        self.roll = get_roll_from_times(data,times,section)

        self.length = self.roll.shape[1]-1


        self.set_key_list(data,section,times)
        self.set_key_profile_list(data,section)

        self.crop(note_range)

        return

    def set_key_list(self,data,section,times):
        if section is None:
            section = [0,self.end_time]

        key_sigs = data.key_signature_changes

        prev_key = 0
        keys_section = []
        times_section = []

        for key_sig in key_sigs:
            key = key_sig.key_number
            time = key_sig.time
            if time < section[0]:
                prev_key = key
            elif time==section[0]:
                keys_section +=[key]
                times_section += [time]
            else: #time > section[0]
                if keys_section == [] and times_section==[]:
                    keys_section +=[prev_key]
                    times_section += [section[0]]
                if time <= section[1]:
                    keys_section +=[key]
                    times_section += [min(time,section[1])]
                #if time > section[1], do nothing

        self.key_list_times = list(zip(keys_section,times_section))

        key_list = []

        for key, time in zip(keys_section,times_section):
            new_time = np.argmin(np.abs(times-time))
            key_list += [(key,new_time)]


        self.key_list = key_list


    def set_key_profile_list(self,data,section):

        key_profiles=[]
        note_range = self.note_range
        roll = (data.get_piano_roll()>0).astype(int)

        if section is None:
            section = [0,self.end_time]

        if self.key_list_times == []:
            key_list = [(0,0)]
        else:
            key_list = self.key_list_times

        times = [x[1] for x in key_list]
        times += [section[1]]

        for time1,time2 in zip(times[:-1],times[1:]):
            idx1 = int(round(time1*100))
            idx2 = int(round(time2*100))

            key_profile = np.sum(roll[:,idx1:idx2],axis=1)/float(idx2-idx1)
            key_profiles += [key_profile]


        self.key_profiles_list = key_profiles

    def crop(self,note_range):
        if self.note_range != note_range:
            old_note_range = self.note_range
            roll = self.roll
            min1 = old_note_range[0]
            max1 = old_note_range[1]
            min2 = note_range[0]
            max2 = note_range[1]

            key_profiles_cropped = []

            if min1<min2:
                new_roll = roll[min2-min1:,:]
            else:
                new_roll = np.append(np.zeros([min1-min2,roll.shape[1]]),roll,0)

            if max1<=max2:
                new_roll = np.append(new_roll,np.zeros([max2-max1,roll.shape[1]]),0)
            else:
                new_roll = new_roll[:-(max1-max2),:]

            #Crop key profiles
            for k in self.key_profiles_list:
                if min1<min2:
                    new_k = k[min2-min1:]
                else:
                    new_k = np.append(np.zeros([min1-min2]),k,0)

                if max1<=max2:
                    new_k = np.append(new_k,np.zeros([max2-max1]),0)
                else:
                    new_k = new_k[:-(max1-max2)]
                key_profiles_cropped+=[new_k]

            self.roll = new_roll
            self.key_profiles_list = key_profiles_cropped
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


    def cut(self,roll,len_chunk,keep_padding=True,as_list=False,with_keys=False):
        #Returns the roll cut in chunks of len_chunk elements, as well as
        #the list of lengths of the chunks
        #The last element is zero-padded to have len_chunk elements


        if keep_padding:
            size = roll.shape[1]
        else:
            size = self.length

        if with_keys:
            key_matrix = self.get_key_profile_matrix()

        if as_list:
            roll_cut = []
            lengths = []
            if with_keys:
                keys = []
        else:
            n_chunks = int(np.ceil(float(size)/len_chunk))
            roll_cut = np.zeros([n_chunks,roll.shape[0],len_chunk])
            lengths = np.zeros([n_chunks])
            if with_keys:
                keys = np.zeros([n_chunks,roll.shape[0],len_chunk])

        j = 0
        n = 0
        length = self.length
        while j < size:

            if j + len_chunk < size:
                if as_list:
                    roll_cut += [roll[:,j:j+len_chunk]]
                    lengths += [min(length,len_chunk)]
                    if with_keys:
                        keys += [key_matrix[:,j:j+len_chunk]]
                else:
                    roll_cut[n]= roll[:,j:j+len_chunk]
                    lengths[n] = min(length,len_chunk)
                    if with_keys:
                        keys[n]= key_matrix[:,j:j+len_chunk]
                j += len_chunk
                n += 1

                length = max(0, length-len_chunk)
            else : #Finishing clause : zero-pad the remaining

                if as_list:
                    roll_cut += [np.pad(roll[:,j:size],pad_width=((0,0),(0,len_chunk-(size-j))),mode='constant')]
                    lengths += [min(length,len_chunk)]
                    if with_keys:
                        keys += [np.pad(key_matrix[:,j:size],pad_width=((0,0),(0,len_chunk-(size-j))),mode='constant')]
                else:
                    roll_cut[n,:,:]= np.pad(roll[:,j:size],pad_width=((0,0),(0,len_chunk-(size-j))),mode='constant')
                    lengths[n] = min(length,len_chunk)
                    if with_keys:
                        keys[n,:,:]= np.pad(key_matrix[:,j:size],pad_width=((0,0),(0,len_chunk-(size-j))),mode='constant')

                j += len_chunk


        outputs = [roll_cut, lengths] if not with_keys else [roll_cut, lengths,keys]


        return outputs

    def transpose(self,diff):
        #Returns a copy of self, transposed of diff semitones
        #diff can be positive or negative
        pr_trans = copy.deepcopy(self)
        roll = self.roll
        if diff<0:
            pr_trans.roll = np.append(roll[-diff:,:],np.zeros([-diff,roll.shape[1]]),0)
            new_profile_list = []
            for profile in self.key_profiles_list:
                new_profile_list += [np.append(profile[-diff:],np.zeros([-diff]))]
            pr_trans.key_profiles_list = new_profile_list
        elif diff>0:
            pr_trans.roll = np.append(np.zeros([diff,roll.shape[1]]),roll[:-diff,:],0)
            new_profile_list = []
            for profile in self.key_profiles_list:
                new_profile_list += [np.append(np.zeros([diff]),profile[:-diff])]
            pr_trans.key_profiles_list = new_profile_list
        #if diff == 0 : do nothing

        pr_trans.key = (self.key+diff)%12
        pr_trans.key_list = [((key+diff)%12,time) for (key,time) in self.key_list]

        return pr_trans


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

    def get_key_profile_matrix(self):
        key_list = self.key_list

        roll = self.roll
        shape = roll.shape
        length = min(roll.shape[1],self.length+1) #Allow 1 more timesteps just in case

        if key_list == []:
            key_list = [(0,0)]

        times = [max(0,x[1]) for x in key_list]
        times += [length]
        key_profile_matrix = np.zeros(shape)
        for time1,time2,key_profile in zip(times[:-1],times[1:],self.key_profiles_list):
            key_profile_repeat = np.tile(key_profile,(time2-time1,1)).transpose()
            key_profile_matrix[:,time1:time2]=key_profile_repeat

        return key_profile_matrix




class PianorollBeats(Pianoroll):

    def make_from_file(self,filename,gt_beats=False,beat_subdiv=[0.0,1.0/4,1.0/3,1.0/2,2.0/3,3.0/4],section=None,note_range=[0,128],key_method='main',with_onsets=False):
        midi_data = pm.PrettyMIDI(filename)
        beats_filename = filename.replace('.mid','_b_gt.csv') if gt_beats else filename.replace('.mid','_b_est.csv')
        beats = np.loadtxt(beats_filename)
        self.make_from_pm(midi_data,beats,beat_subdiv,section,note_range,key_method,with_onsets)
        self.name = os.path.splitext(os.path.basename(filename))[0]
        return

    def make_from_pm(self,data,beats,beat_subdiv=[0.0,1.0/4,1.0/3,1.0/2,2.0/3,3.0/4],section=None,note_range=[0,128],key_method='main',with_onsets=False):


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

        # Make step times from beats and beat_subdiv
        n_subdiv=len(beat_subdiv)
        n_beats = len(beats)-1 #Only take beat intervals that have an end

        beat_duration = beats[1:]-beats[:-1]
        offset_to_beat = np.tile(beat_subdiv,n_beats)*np.repeat(beat_duration,n_subdiv)
        times = np.repeat(beats[:-1],n_subdiv) + offset_to_beat

        total_duration = data.get_piano_roll().shape[1]/100.0
        end_time = min(section[1],total_duration) if section is not None else total_duration
        self.end_time = end_time

        self.roll = get_roll_from_times(data,times,section,with_onsets)

        self.length = self.roll.shape[1]-1

        self.set_key_list(data,section,times)
        self.set_key_profile_list(data,section)

        self.crop(note_range)

        return

    def set_key_list(self,data,section,times):
        if section is None:
            section = [0,self.end_time]

        key_sigs = data.key_signature_changes

        prev_key = 0
        keys_section = []
        times_section = []

        for key_sig in key_sigs:
            key = key_sig.key_number
            time = key_sig.time
            if time < section[0]:
                prev_key = key
            elif time==section[0]:
                keys_section +=[key]
                times_section += [time]
            else: #time > section[0]
                if keys_section == [] and times_section==[]:
                    keys_section +=[prev_key]
                    times_section += [section[0]]
                if time <= section[1]:
                    keys_section +=[key]
                    times_section += [min(time,section[1])]
                #if time > section[1], do nothing

        self.key_list_times = list(zip(keys_section,times_section))

        key_list = []

        for key, time in zip(keys_section,times_section):
            new_time = np.argmin(np.abs(times-time))
            key_list += [(key,new_time)]

        self.key_list = key_list

def get_roll_from_times(midi_data,times,section=None,with_onsets=False):
    # quant_piano_roll = midi_data.get_piano_roll(fs=500,times=times)
    # quant_piano_roll = (quant_piano_roll>=7).astype(int)
    roll = np.zeros([128,len(times)])

    for instr in midi_data.instruments:
        for note in instr.notes:
            start = np.argmin(np.abs(times-note.start))
            end = np.argmin(np.abs(times-note.end))
            if start == end:
                end = start+1
            roll[note.pitch,start:end]=1

            if with_onsets:
                roll[note.pitch,onset_idx] = 2


    if not section == None:
        begin = section[0]
        end = section[1]
        assert begin < end
        begin_index = np.argmin(np.abs(begin-times))
        end_index = np.argmin(np.abs(end-times))
        roll = roll[:,begin_index:end_index]

    return roll

def get_quant_piano_roll(midi_data,fs=4,section=None,with_onsets=False):
    # DEPRECATED!!!
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

    if with_onsets:
        for instr in data.instruments:
            for note in instr.notes:
                onset_idx = int(round(note.start*fs))
                quant_piano_roll[note.pitch,onset_idx] = 2


    if not section == None:
        begin = section[0]
        end = section[1]
        assert begin < end

        begin_index = int(round(midi_data.time_to_tick(begin)/PPQ*fs))
        end_index = int(round(midi_data.time_to_tick(end)/PPQ*fs))
        quant_piano_roll = quant_piano_roll[:,begin_index:end_index]

    return quant_piano_roll

def get_event_roll(midi_data,section=None):
    # DEPRECATED!!!
    steps = np.unique(midi_data.get_onsets())

    #Remove onsets that are within 50ms of each other (keep first one only)
    diff = steps[1:] - steps[:-1]
    close = diff<0.05
    while np.any(close):
        to_keep = np.where(np.logical_not(close))
        steps = steps[to_keep[0]+1]
        diff = steps[1:] - steps[:-1]
        close = diff<0.05


    for s1,s2 in zip(steps[:-1],steps[1:]):
        if s2-s1 < 0.05:
            print(s1, s2, s2-s1)
            print(round(s1*20)/20, round(s2*20)/20)

    pr = get_piano_roll_from_times(midi_data,steps,section)

    return pr, steps





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

def check_correct_onsets(roll):
    diff = roll[:,1:]-roll[:,:-1]

    if np.all(np.logical_or.reduce((diff==2,diff==-1,diff==0))):
        pass
    else:
        incorrect= np.where(np.logical_not(np.logical_or.reduce((diff==2,diff==-1,diff==0))))
        for idx in zip(incorrect[0],incorrect[1]):
            print(idx)
        import matplotlib.pyplot as plt
        plt.imshow(pr.roll,aspect='auto',origin='lower')
        plt.show(block=[bool])




# pr=PianorollBeats()
# pr.make_from_file('data/piano-midi-ttv-20p/test/bor_ps6.mid',gt_beats=False,section=[0,30],note_range=[21,109])
#
# import matplotlib.pyplot as plt
#
# plt.imshow(pr.roll,aspect='auto',origin='lower')
# ax = plt.gca()
# ax.set_xticks(np.arange(0,pr.roll.shape[1]))
# ax.grid(linestyle='-', linewidth=0.5)
# plt.show()

# folder = 'data/piano-midi-ttv-20p/train'
# for fn in os.listdir(folder):
#     if fn.endswith('.mid') and not fn.startswith('.'):
#         filename = os.path.join(folder,fn)
#         print filename
#         pr = Pianoroll()
#         pr.make_from_file(filename,'time',note_range=[21,109],with_onsets=True)
#         check_correct_onsets(pr.roll)


# pr = Pianoroll()
# pr.roll = np.array([[1,1,2,2,3,3],[4,4,5,5,6,6],[1,1,2,2,3,3]])
# pr.length = 6
# pr.note_range=[3,6]
# print pr.split_pitchwise(1)[0]
