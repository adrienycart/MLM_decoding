import os
import numpy as np
import pretty_midi as pm
from random import shuffle
import pickle as pickle
from datetime import datetime
import copy
from mlm_training.pianoroll import Pianoroll
# from pianoroll import Pianoroll
from tqdm import tqdm

from queue import Queue
from threading import Thread


class ThreadedDataset:
    def __init__(self,folder,timestep_type,chunks_len,rand_transp=True,note_range=[21,109]):
        self.train = []
        self.valid = []
        self.valid_rolls = []
        self.valid_lens = []

        self.timestep_type = timestep_type
        self.note_range = note_range
        self.rand_transp=rand_transp
        self.chunks_len = chunks_len


        self.init_data_list(folder)
        self.set_valid_data()


    def init_data_list(folder):
        for subset in ['train','valid']:
            file_list = []

            for fn in os.listdir(os.path.join(folder,subset)):
                if (fn.endswith('.mid') or fn.endswith('.midi')) and not fn.startswith('.'):
                    file_list += [os.path.join(folder,subset,fn)]
            setattr(self,subset,file_list)

    def load_one_midi(self,filename):
        midi_data = pm.PrettyMIDI(filename)
        piano_roll = Pianoroll()
        piano_roll.make_from_pm(midi_data,self.timestep_type,None,self.note_range)
        chunks, chunks_len = piano_roll.cut(len_chunk,keep_padding=False,as_list=True)
        return chunks, chunks_len

    def set_valid_data():

        data = []
        lengths = []

        for filename in self.valid:
            chunks, chunks_len = self.load_one_midi(filename)
            data.extend(chunks)
            lengths.extend(chunks_len)

        self.valid_rolls = np.array(data)
        self.valid_lens  = np.array(lengths)

    def init_queue_thread(self,batch_size):
        self.queue = Queue(maxsize=batch_size*10)
        thread = threading.Thread(target=self.enqueue_piano_rolls, args=())


    def enqueue_piano_rolls():
        for filename in self.train:
            chunks, chunks_len = self.load_one_midi(filename)
            for chunk, chunk_len in zip(chunks, chunks_len):
                self.queue.put([chunk, chunk_len])


    def shuffle_one(self,subset):
        data = getattr(self,subset)
        shuffle(data)







class Dataset:
    """Classe representing the dataset."""

    def __init__(self,rand_transp=False):
        self.train = []
        self.test = []
        self.valid = []

        self.note_range = [0,128]
        self.max_len = None
        self.rand_transp=rand_transp

    def walkdir(self,folder):
        for fn in os.listdir(folder):
            if (fn.endswith('.mid') or fn.endswith('.midi')) and not fn.startswith('.'):
                yield fn


    def load_data_one(self,folder,subset,timestep_type,max_len=None,note_range=[0,128],length_of_chunks=None,key_method='main'):
        dataset = []
        subfolder = os.path.join(folder,subset)

        #Set up progress bar
        filecounter = 0
        for filepath in self.walkdir(subfolder):
            filecounter += 1
        print("Now loading: "+subset.upper())
        pbar = tqdm(self.walkdir(subfolder), total=filecounter, unit="files")
        for fn in pbar:
            pbar.set_postfix(file=fn[:10], refresh=False)
            filename = os.path.join(subfolder,fn)
            # print filename
            midi_data = pm.PrettyMIDI(filename)
            if length_of_chunks == None:
                piano_roll = Pianoroll()
                if max_len == None:
                    piano_roll.make_from_pm(midi_data,timestep_type,None,note_range,key_method)
                else:
                    piano_roll.make_from_pm(midi_data,timestep_type,[0,max_len],note_range,key_method)
                piano_roll.name = os.path.splitext(os.path.basename(filename))[0]
                dataset += [piano_roll]
            else :
                if max_len == None:
                    end_file = midi_data.get_piano_roll().shape[1]/100.0
                else :
                    end_file = max_len
                begin = 0
                end = 0
                i = 0
                pr_list = []
                while end < end_file:
                    end = min(end_file,end+length_of_chunks)
                    piano_roll = Pianoroll()
                    piano_roll.make_from_pm(midi_data,timestep_type,[begin,end],note_range,key_method)
                    piano_roll.name = os.path.splitext(os.path.basename(filename))[0]+"_"+str(i)
                    pr_list += [piano_roll]
                    begin = end
                    i += 1
                dataset += pr_list

        if subset in ["train","valid","test"]:
            setattr(self,subset,dataset)
        return dataset

    def load_data(self,folder,timestep_type,max_len=None,note_range=[0,128],length_of_chunks=None,key_method='main'):
        self.note_range = note_range
        for subset in ["train","valid","test"]:
            self.load_data_one(folder,subset,timestep_type,max_len,note_range,length_of_chunks,key_method)
        # self.zero_pad()
        print("Dataset loaded ! "+str(datetime.now()))

    def load_data_custom(self,folder,train,valid,test,timestep_type,max_len=None,note_range=[0,128],length_of_chunks=None):
        self.note_range = note_range

        for subset in train:
            self.train += self.load_data_one(folder,subset,timestep_type,max_len,note_range,length_of_chunks)
        for subset in valid:
            self.valid += self.load_data_one(folder,subset,timestep_type,max_len,note_range,length_of_chunks)
        for subset in test:
            self.test += self.load_data_one(folder,subset,timestep_type,max_len,note_range,length_of_chunks)

        print("Dataset loaded ! "+str(datetime.now()))

    def get_n_files(self,subset):
        return len(getattr(self,subset))
    def get_n_notes(self):
        return self.note_range[1]-self.note_range[0]
    def get_len_files(self):
        return self.max_len

    def get_dataset(self,subset,with_names=False,with_key_masks=False):
        #Outputs an array containing all the piano-rolls (3D-tensor)
        #and the list of the actual lengths of the piano-rolls
        pr_list = getattr(self,subset)
        n_files = len(pr_list)
        len_file = pr_list[0].roll.shape[1]
        n_notes = self.get_n_notes()

        dataset = np.zeros([n_files,n_notes,len_file])
        lengths = np.zeros([n_files],dtype=int)
        if with_names:
            names = []

        for i, piano_roll in enumerate(pr_list):
            roll = piano_roll.roll
            dataset[i] = roll
            lengths[i] = piano_roll.length
            if with_names:
                names += [piano_roll.name]

        output = [dataset, lengths]
        if with_names:
            output += [names]
        if with_key_masks:
            output += [key_masks]
            #Zero-pad the key_lists
            max_len = max(list(map(len,key_lists)))
            key_lists_array = np.zeros([n_files,max_len])
            for i,key_list in enumerate(key_lists):
                key_lists_array[i,:len(key_list)]=key_list
                key_lists_array[i,len(key_list):]=key_list[-1]
            output += [key_lists_array]
        return output

    def get_dataset_chunks(self,subset,len_chunk):
        #Outputs an array containing all the pieces cut in chunks (4D-tensor)
        #and a list of lists for the lengths
        pr_list = getattr(self,subset)
        n_files = len(pr_list)
        len_file = pr_list[0].roll.shape[1]
        n_notes = self.get_n_notes()
        n_chunks = int(np.ceil(float(len_file)/len_chunk))

        dataset = np.zeros([n_files,n_chunks,n_notes,len_chunk])
        lengths = np.zeros([n_files,n_chunks])
        i = 0
        while i<n_files:
            piano_roll = pr_list[i]
            chunks, chunks_len = piano_roll.cut(len_chunk)
            dataset[i] = chunks
            lengths[i] = chunks_len
            i += 1

        return dataset[:,:,:-1], dataset[:,:,1:], lengths

    def get_dataset_chunks_no_pad(self,subset,len_chunk,with_keys=False):
        #Outputs an array containing all the pieces cut in chunks (3D-tensor)
        #and a list for the lengths
        pr_list = getattr(self,subset)
        n_files = len(pr_list)
        len_file = pr_list[0].roll.shape[1]
        n_notes = self.get_n_notes()

        dataset = []
        lengths = []
        if with_keys:
            keys = []

        i = 0
        while i<n_files:
            piano_roll = pr_list[i]
            if self.rand_transp:
                transp = np.random.randint(-7,6)
                piano_roll = piano_roll.transpose(transp)
            if with_keys:
                chunks, chunks_len, chunks_keys = piano_roll.cut(piano_roll.roll,len_chunk,keep_padding=False,with_keys = True)
                keys += list(chunks_keys)
            else:
                chunks, chunks_len = piano_roll.cut(piano_roll.roll,len_chunk,keep_padding=False)
            dataset += list(chunks)
            lengths += list(chunks_len)
            i += 1

        if with_keys:
            outputs = [np.asarray(dataset)[:,:,:-1],np.asarray(dataset)[:,:,1:], np.asarray(lengths),np.asarray(keys)[:,:,1:]]
        else:
            outputs = [np.asarray(dataset)[:,:,:-1],np.asarray(dataset)[:,:,1:], np.asarray(lengths)]

        return outputs


    def get_dataset_generator(self,subset,batch_size,len_chunk=None,with_names=False,with_keys=False,check_data=False):
        seq_buff = []
        len_buff = []
        if with_names:
            names_buff = []
        if with_keys:
            keys_buff = []
        pr_list = getattr(self,subset)
        files_left = list(range(len(pr_list)))

        n_notes = self.note_range[1]-self.note_range[0]
        if self.max_len is None:
            self.set_max_len()

        while files_left != [] or len(seq_buff)>=batch_size:
            if len(seq_buff)<batch_size:
                file_index = files_left.pop()
                piano_roll = pr_list[file_index]
                if self.rand_transp:
                    transp = np.random.randint(-7,6)
                    piano_roll = piano_roll.transpose(transp)
                if len_chunk is None:
                    roll= piano_roll.roll
                    length = piano_roll.length
                    seq_buff.append(roll)
                    len_buff.append(length)
                else:
                    if with_keys:
                        chunks, chunks_len,chunks_keys = piano_roll.cut(piano_roll.roll,len_chunk,keep_padding=False,as_list=True,with_keys=True)
                        if check_data:
                            # print(piano_roll.name)
                            idx = self.check_data(chunks,chunks_len)

                            chunks = [chunks[i] for i in idx]
                            chunks_len = [chunks_len[i] for i in idx]
                            chunks_keys = [chunks_keys[i] for i in idx]
                        seq_buff.extend(chunks)
                        len_buff.extend(chunks_len)
                        keys_buff.extend(chunks_keys)
                    else:
                        chunks, chunks_len = piano_roll.cut(piano_roll.roll,len_chunk,keep_padding=False,as_list=True)
                        seq_buff.extend(chunks)
                        len_buff.extend(chunks_len)
                    if with_names:
                        n_chunks = len(chunks)
                        name = piano_roll.name
                        names_buff.extend([name+'_ch'+str(i) for i in range(n_chunks)])
            else:
                output_roll = np.array(seq_buff[:batch_size])
                output = [output_roll[:,:,:-1],output_roll[:,:,1:],np.array(len_buff[:batch_size])-1]
                del seq_buff[:batch_size]
                del len_buff[:batch_size]

                if with_names:
                    output_names = names_buff[:batch_size]
                    output += [output_names]
                    del names_buff[:batch_size]
                if with_keys:
                    output_keys = np.array(keys_buff[:batch_size])
                    # print(output_keys.shape,output_roll.shape)
                    output += [output_keys[:,:,1:]]
                    del keys_buff[:batch_size]


                yield output


    def get_pitchwise_dataset_generator(self,subset,batch_size,window_size,len_chunk=None):
        seq_buff = []
        targ_buff = []
        len_buff = []
        pr_list = getattr(self,subset)
        files_left = list(range(len(pr_list)))

        if self.max_len is None:
            self.set_max_len()

        while files_left != [] or len(seq_buff)>=batch_size:
            if len(seq_buff)<batch_size:
                file_index = files_left.pop()
                piano_roll = pr_list[file_index]

                inputs, targets = piano_roll.split_pitchwise(window_size)

                if len_chunk is None:
                    seq_buff.extend(inputs)
                    targ_buff.extend(targets)
                    len_buff.extend([piano_roll.length]*len(inputs))
                else:
                    for input,target in zip(inputs,targets):
                        chunks_input, chunks_len = piano_roll.cut(input,len_chunk-1,keep_padding=False,as_list=True)
                        chunks_target, chunks_len = piano_roll.cut(target,len_chunk-1,keep_padding=False,as_list=True)
                        seq_buff.extend(chunks_input)
                        targ_buff.extend(chunks_target)
                        len_buff.extend(chunks_len)
            else:
                if len_chunk is None:
                    output_roll = np.zeros([batch_size,2*window_size+1,self.max_len-1])
                    output_targets = np.zeros([batch_size,1,self.max_len-1])
                    for i,(seq,targ) in enumerate(zip(seq_buff[:batch_size],targ_buff[:batch_size])):
                        output_roll[i,:,:seq.shape[1]]=seq
                        output_targets[i,:,:targ.shape[1]]=targ
                    output = (output_roll,output_targets,np.array(len_buff[:batch_size]))
                else:
                    np.array(seq_buff[:batch_size])
                    output = (np.array(seq_buff[:batch_size]),np.array(targ_buff[:batch_size]),np.array(len_buff[:batch_size]))
                del seq_buff[:batch_size]
                del targ_buff[:batch_size]
                del len_buff[:batch_size]
                yield output

    def check_data(self,rolls,lengths,names=None):
        to_keep = []
        if names is None:
            zip_list = zip(rolls,lengths)
        else:
            zip_list = zip(rolls,lengths,names)
        for i,values in enumerate(zip(rolls,lengths)):
            if names is None:
                (roll,length) = values
            else:
                (roll,length,name) = values
                # print(name)

            #When exported, we do -1 to the lengths so this has to be the case here as well
            length = int(length-1)
            pr = roll[:,:length]

            data_extended = np.pad(pr,[[0,0],[1,1]],'constant')
            diff = data_extended[:,1:] - data_extended[:,:-1]
            steady = np.where(np.sum(np.abs(diff),axis=0)==0)[0]
            trans = np.where(np.sum(np.abs(diff),axis=0)!=0)[0]

            steady_ok = False
            trans_ok = False

            try:
                if steady[-1] == pr.shape[1]:
                    steady = steady[:-1]
            except IndexError:
                assert steady.size == 0
            else:
                try:
                    if steady[0] == 0:
                        steady = steady[1:]
                except IndexError:
                    assert steady.size == 0
                else:
                    if steady.size == 0:
                        pass
                    else:
                        steady_ok = True

            try:
                if trans[-1] == pr.shape[1]:
                    trans = trans[:-1]
            except IndexError:
                assert trans.size == 0
            else:
                try:
                    if trans[0] == 0:
                        trans = trans[1:]
                except IndexError:
                    assert trans.size == 0
                else:
                    if trans.size == 0:
                        pass
                    else:
                        trans_ok = True
            # print(pr.shape,length)
            # print(steady_ok,trans_ok)
            # print(steady,trans)
            if steady_ok and trans_ok:
                to_keep += [i]
        return to_keep



    def shuffle_one(self,subset):
        data = getattr(self,subset)
        shuffle(data)


    def __max_len(self,dataset):
        if dataset == []:
            return 0
        else :
            return max([x.length for x in dataset])

    def set_max_len(self):
        max_train = self.__max_len(self.train)
        max_valid = self.__max_len(self.valid)
        max_test = self.__max_len(self.test)
        max_len = max([max_train,max_valid,max_test])
        self.max_len = max_len

    def zero_pad(self):
        if self.max_len is None:
            self.set_max_len()


        for subset in ["train","valid","test"]:
            self.zero_pad_one(subset,self.max_len)


    def zero_pad_one(self,subset,max_len):
        #Zero-padding the dataset
        dataset = getattr(self,subset)
        for piano_roll in dataset:
            piano_roll.zero_pad(max_len)


    def transpose_all_one(self,subset):
        data = getattr(self,subset)
        tr_range = [-7,5]
        new_data = []
        for piano_roll in data:
            for j in range(*tr_range):
                tr_piano_roll = piano_roll.transpose(j)
                new_data += [tr_piano_roll]
        setattr(self,subset,new_data)

    def transpose_all(self):
        print("Transposing train set in every key...")
        #You only augment the training dataset
        for subset in ["train"]: #,"valid","test"]:
            self.transpose_all_one(subset)



def ground_truth(data):
    return data[:,:,1:]




#
# folder = "data/Piano-midi.de/"
# for subfolder in ["train","valid","test"]:
#     subfolder = os.path.join(folder,subfolder)
#     for fn in os.listdir(subfolder):
#         if fn.endswith('.mid') and not fn.startswith('.'):
#             filename = os.path.join(subfolder,fn)
#             midi_data = pm.PrettyMIDI(filename)
#             time_signatures = midi_data.time_signature_changes
#             if not (time_signatures[0].denominator == 4 or (time_signatures[0].numerator == 4 and time_signatures[0].numerator == 2)):
#                 print filename
#                 print midi_data.time_signature_changes
#                 print midi_data.get_end_time()



# data = Dataset()
# data.load_data('data/test_dataset/',
#         timestep_type='quant',max_len=None,note_range=[21,109])
# #
# # # for pr in data.test:
# #     # print pr.name
# #
# # data_gen = data.get_dataset_generator('test',2,len_chunk = None)
# data_gen = data.get_dataset_generator('test',4,len_chunk = 300)
#
#
# for batch,target,lens in data_gen:
#     # pass
#     print(batch.shape, target.shape)
#     print(lens)
# for pr in data.test:
#     print pr.name
#
# dataset,lengths = data.get_dataset_chunks_no_pad('test',50)
# ptr = 0
# print "================"
# while ptr+3<dataset.shape[0]:
#     print lengths[ptr:ptr+3]
#     ptr +=3
# for pr in data.test:
#     print pr.name

# print 'finish'

# # data.load_data_custom('data/Piano-midi-sorted',train=['albeniz','borodin'],valid=['clementi'],test=['grieg'],
# #         fs=4,max_len=15,note_range=[21,109],quant=True,length_of_chunks=None)
# # print data.train
# for pr in data.train:
#     print(pr.name)
#     print(pr.length)



#split_files("data/dummy_midi_data_poly_upwards/")
#unsplit_files("dummy_midi_data_poly/")

# liste = get_chord_counter('dummy_midi_data_poly/train')
# nums = [ x[1] for x in liste]
# print max(nums)
# print min(nums)
# print max(nums)-min(nums)


# for fn in os.listdir("data/Piano-midi.de"):
#     path = os.path.join("data/Piano-midi.de",fn)
#     if os.path.isdir(path):
#         split_files(path)
#
# safe_mkdir("data/Piano-midi.de/train")
# safe_mkdir("data/Piano-midi.de/valid")
# safe_mkdir("data/Piano-midi.de/test")
#
# for fn in os.listdir("data/Piano-midi.de"):
#     folder = os.path.join("data/Piano-midi.de",fn)
#     if os.path.isdir(folder):
#         train_path = os.path.join(folder,"train/")
#         valid_path = os.path.join(folder,"valid/")
#         test_path = os.path.join(folder,"test/")
#
#         move_files(os.listdir(train_path),train_path,"data/Piano-midi.de/train")
#         move_files(os.listdir(valid_path),valid_path,"data/Piano-midi.de/valid")
#         move_files(os.listdir(test_path),test_path,"data/Piano-midi.de/test")
