import os
import numpy as np
import pretty_midi as pm
import random
import pickle as pickle
from datetime import datetime
import copy
from tqdm import tqdm
from dataMaps import DataMaps



class DatasetMaps:
    """Classe representing the dataset."""

    def __init__(self,rand_transp=False):
        self.train = []
        self.test = []
        self.valid = []

        self.note_range = [0,128]
        self.max_len = None
        self.acoustic_model = ""
        self.rand_transp=rand_transp


    def walkdir(self,folder):
        for fn in os.listdir(folder):
            if fn.endswith('.mid') and not fn.startswith('.'):
                yield fn

    def get_list_of_dataMaps(self,subfolder,timestep_type,max_len=None,length_of_chunks=None,method='step'):
        dataset = []


        #Set up progress bar
        filecounter = 0
        for filepath in self.walkdir(subfolder):
            filecounter += 1
        print(("Now loading: "+os.path.split(subfolder)[-1].upper()))
        pbar = tqdm(self.walkdir(subfolder), total=filecounter, unit="files")

        for fn in pbar:
            pbar.set_postfix(file=fn[9:19], refresh=False)
            filename = os.path.join(subfolder,fn)

            if length_of_chunks == None:
                data = DataMaps()
                if max_len == None:
                    data.make_from_file(filename,timestep_type,None,method,acoustic_model=self.acoustic_model)
                else:
                    data.make_from_file(filename,timestep_type,[0,max_len],method,acoustic_model=self.acoustic_model)
                data.name = os.path.splitext(os.path.basename(filename))[0]
                dataset += [data]
            else :
                #Cut each file in chunks of 'length_of_chunks' seconds
                #Make a new dataMaps for each chunk.
                data_whole = DataMaps()
                data_whole.make_from_file(filename,timestep_type,None,method,acoustic_model=self.acoustic_model)
                if max_len == None:
                    end_file = data_whole.duration
                else :
                    end_file = max_len
                begin = 0
                end = 0
                i = 0
                data_list = []
                while end < end_file:
                    end = min(end_file,end+length_of_chunks)
                    data = data_whole.copy_section([begin,end])
                    data.name = os.path.splitext(os.path.basename(filename))[0]+"_"+str(i)
                    data_list += [data]
                    begin = end
                    i += 1
                dataset += data_list
        return dataset

    def load_data(self,folder,timestep_type,max_len=None,length_of_chunks=None,method='avg',subsets=['valid','test'],acoustic_model="kelz"):
        if acoustic_model == 'benetos':
            self.note_range = [21,109]
        elif acoustic_model == 'kelz':
            self.note_range = [21,109]
        elif acoustic_model == 'bittner':
            # self.note_range = [24,97]
            self.note_range = [21,109]

        self.acoustic_model = acoustic_model

        for subset in subsets:
            subfolder = os.path.join(folder,subset)
            data_list = self.get_list_of_dataMaps(subfolder,timestep_type,max_len,length_of_chunks,method)
            setattr(self,subset,data_list)

        print("Dataset loaded ! "+str(datetime.now()))

    def get_n_files(self,subset):
        return len(getattr(self,subset))
    def get_n_notes(self):
        return self.note_range[1]-self.note_range[0]
    def get_len_files(self):
        return self.max_len


    def get_dataset_chunks_no_pad(self,subset,len_chunk):
        #Outputs an array containing all the pieces cut in chunks (3D-tensor)
        #and a list for the lengths
        data_list = getattr(self,subset)
        n_files = len(data_list)
        n_notes = self.get_n_notes()

        inputs = []
        targets = []
        lengths = []

        i = 0
        while i<n_files:
            data = data_list[i]

            input_chunks,target_chunks, chunks_len = data.cut(len_chunk,keep_padding=False,as_list=True)
            inputs += input_chunks
            targets += target_chunks
            lengths += chunks_len
            i += 1

        return np.asarray(inputs), np.asarray(targets),np.asarray(lengths)

    def get_dataset_generator(self,subset,batch_size,len_chunk=None):
        seq_buff = []
        targets_buff = []
        len_buff = []
        data_list = getattr(self,subset)
        files_left = list(range(len(data_list)))

        n_notes = self.note_range[1]-self.note_range[0]
        if self.max_len is None:
            self.set_max_len()

        while files_left != [] or len(seq_buff)>=batch_size:
            if len(seq_buff)<batch_size:
                file_index = files_left.pop()
                data = data_list[file_index]

                if self.rand_transp:
                    transp = np.random.randint(-3,3)
                    data = data.transpose(transp)

                if len_chunk is None:
                    roll= data.input
                    target = data.target
                    length = data.length
                    seq_buff.append(roll)
                    len_buff.append(length)
                    targets_buff.append(target)
                else:
                    chunks_in,chunks_tar, chunks_len = data.cut(len_chunk,keep_padding=False,as_list=True)
                    seq_buff.extend(chunks_in)
                    targets_buff.extend(chunks_tar)
                    len_buff.extend(chunks_len)
            else:
                if len_chunk is None:
                    output_seq = np.zeros([batch_size,n_notes,self.max_len])
                    output_tar = np.zeros([batch_size,n_notes,self.max_len])
                    for i,(seq,tar) in enumerate(zip(seq_buff[:batch_size],targets_buff[:batch_size])):
                        output_seq[i,:,:seq.shape[1]]=seq
                        output_tar[i,:,:tar.shape[1]]=tar
                    output = (output_seq,output_tar,np.array(len_buff[:batch_size]))
                else:
                    output_seq = np.array(seq_buff[:batch_size])
                    output_tar = np.array(targets_buff[:batch_size])
                    output = (output_seq,output_tar,np.array(len_buff[:batch_size]))
                del seq_buff[:batch_size]
                del targets_buff[:batch_size]
                del len_buff[:batch_size]
                yield output


    def shuffle_one(self,subset):
        data = getattr(self,subset)
        random.shuffle(data)


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


    def zero_pad(self, subset='all'):
        #Zero-padding the dataset
        if subset == 'all':
            max_len = get_len_files()
            for subset in ["train","valid","test"]:
                self.zero_pad_one(subset,max_len)
        else:
            max_len = self.__max_len(getattr(self, subset))
            self.zero_pad_one(subset,max_len)


    def zero_pad_one(self,subset,max_len):
        #Zero-padding the dataset
        dataset = getattr(self,subset)
        for data in dataset:
            data.zero_pad('input',max_len)
            data.zero_pad('target',max_len)
        return

    def convert_note_to_time(self,subset,piano_rolls,fs,max_len):
        #Convert a set of piano_rolls (3D-tensor) from note-based to time-based time steps
        dataset = getattr(self,subset)
        assert len(dataset) == piano_rolls.shape[0]
        piano_rolls_time = np.zeros([piano_rolls.shape[0],piano_rolls.shape[1],int(round(max_len*fs))])
        for i in range(piano_rolls.shape[0]):
            data = dataset[i]
            roll = piano_rolls[i]
            piano_rolls_time[i] = data.convert_note_to_time(roll,fs,max_len)
        return piano_rolls_time

    def convert_time_to_note(self,subset,piano_rolls,fs,max_len):
        #Convert a set of piano_rolls (3D-tensor) from time-based to note-based time steps
        dataset = getattr(self,subset)
        assert len(dataset) == piano_rolls.shape[0]
        piano_rolls_note = []
        for i in range(piano_rolls.shape[0]):
            data = dataset[i]
            roll = piano_rolls[i]
            piano_rolls_note += [data.convert_time_to_note(roll,fs,max_len)]
        #Zero-pad all the note-based piano rolls
        max_len = max([x.shape[1] for x in piano_rolls_note])
        piano_rolls_note_padded = np.zeros([piano_rolls.shape[0],piano_rolls.shape[1],max_len])
        for i in range(piano_rolls.shape[0]):
            roll = piano_rolls_note[i]
            roll_padded = np.pad(roll,pad_width=((0,0),(0,max_len-roll.shape[1])),mode='constant')
            piano_rolls_note_padded[i] = roll_padded
        return piano_rolls_note_padded


    def make_norm_data_name(self,folder,quant,method):
        if quant:
            return os.path.join(folder,'norm_data_quant_'+method+'.p')
        else:
            return os.path.join(folder,'norm_data_unquant.p')

    def normalize_all(self,folder,quant,method):
        #Normalize each dimension by substracting the mean
        #and dividing by the variance over test dataset
        #mean and var are pre-computed as pickle files
        norm_data = pickle.load(open(self.make_norm_data_name(folder,quant,method), "rb"))
        mean = norm_data['mean']
        var = norm_data['var']

        for subset in ["train","valid","test"]:
            self.normalize_one(subset,mean,var)

    def normalize_one(self,subset,mean,var):
        dataset = getattr(self,subset)
        for data in dataset:
            data.normalize_input(mean,var)
        return

    def write_norm_data(self,path,name):
        #To compute the mean and var of the dataset and write it in a pickle file
        inputs, targets, lengths = self.get_dataset('train')
        mean = np.mean(inputs,axis=2)
        mean = np.mean(mean,axis=0)

        var = np.var(inputs,axis=2)
        var = np.var(var,axis=0)
        var += np.full(var.shape,np.finfo(float).eps)

        norm_data = {}
        norm_data['mean'] = mean
        norm_data['var'] = var

        import pickle as pickle
        pickle.dump(norm_data, open(os.path.join(path,name), "wb"))
        return




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


data = DatasetMaps(rand_transp=True)
data.load_data('data/outputs_default_config_split','quant',max_len=30,subsets=['valid'],acoustic_model="bittner")
inputs, targets, lens =data.get_dataset_chunks_no_pad('valid',100)
#
# print(inputs.shape, targets.shape, lens.shape)
data_gen = data.get_dataset_generator('valid',10)
# # #
# # #
for inputs,target,lens in data_gen:
#     # pass
    print(inputs.shape, target.shape)
#
#     inputs_1 = inputs[target==1]
#
#     print(np.max(inputs_1),np.mean(inputs_1),np.std(inputs_1))


#######
######
# RUN THIS JUST TO BE SURE, then copy the new data to the server
######
#######

# data = DatasetMaps()
# path_data = 'data/Config1/fold1'
# # path_data = 'data/test_dataset'
# data.load_data(path_data,
#         fs=4,max_len=None,note_range=[21,109],quant=True,length_of_chunks=30,posteriogram=True,annot_path='corresp_dataset/full_dataset.p')
# # lengths = []
# for pr in data.train:
#     lengths += [pr.length]
#
# import matplotlib.pyplot as plt
#
# plt.hist(lengths,bins=25,normed=True)
# plt.show()

# configs = ["Config1","Config2"]
# folds = ["fold1", "fold2", "fold3", "fold4"]
# methods = ['avg','step','exp']
#
# print "##################"
# print "WRITING NORM DATA"
# print "##################"
#
# for config in configs:
#     for fold in folds:
#         path = os.path.join('data',config,fold)
#         data = DatasetMaps()
#         data.load_data(path,
#                 fs=100,max_len=None,note_range=[21,109],quant=False,length_of_chunks=30,posteriogram=True)
#         data.write_norm_data(path,"norm_data_unquant.p")
#
# for config in configs:
#     for fold in folds:
#         for method in methods:
#             path = os.path.join('data',config,fold)
#             data = DatasetMaps()
#             data.load_data(path,
#                     fs=100,max_len=None,note_range=[21,109],quant=False,length_of_chunks=30,posteriogram=True,method=method)
#             data.write_norm_data(path,"norm_data_quant_"+method+".p")
