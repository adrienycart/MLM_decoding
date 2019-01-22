import os
import numpy as np
import pretty_midi as pm
import random
import cPickle as pickle
from datetime import datetime
import copy
from dataMaps import DataMaps


class DatasetMaps:
    """Classe representing the dataset."""

    def __init__(self):
        self.train = []
        self.test = []
        self.valid = []

        self.note_range = [0,128]
        self.max_len = 0


    def get_list_of_dataMaps(self,subfolder,fs,max_len=None,note_range=[0,128],quant=False,length_of_chunks=None,posteriogram=False,method='avg',augm=False,annot=None):
        dataset = []
        print "Augm",augm
        for fn in os.listdir(subfolder):
            if fn.endswith('.mid') and not fn.startswith('.'):
                filename = os.path.join(subfolder,fn)
                annot_data = annot[os.path.basename(filename)]
                print filename
                if augm:
                    if 'train' in subfolder:
                        transps = [-2, -1,0,1,2]
                    else:
                        transps = [0]
                else :
                    transps = [None]
                for transp in transps:
                    print "transp",transp
                    if length_of_chunks == None:
                        data = DataMaps()
                        if max_len == None:
                            data.make_from_file(filename,fs,None,note_range,quant,posteriogram,method,transp,annot_data)
                        else:
                            data.make_from_file(filename,fs,[0,max_len],note_range,quant,posteriogram,method,transp,annot_data)
                        data.name = os.path.splitext(os.path.basename(filename))[0]
                        dataset += [data]
                    else :
                        #Cut each file in chunks of 'length_of_chunks' seconds
                        #Make a new dataMaps for each chunk.
                        data_whole = DataMaps()
                        data_whole.make_from_file(filename,fs,None,note_range,quant,posteriogram,method,transp,annot_data)
                        if max_len == None:
                            end_file = data_whole.corresp[-1,0]
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

    def load_data_train_valid(self,folder,fs,max_len=None,note_range=[0,128],quant=False,length_of_chunks=None,posteriogram=False,method='avg',augm=False,annot=None):
        subfolder = os.path.join(folder,'train')
        print "Augm", augm
        dataset = self.get_list_of_dataMaps(subfolder,fs,max_len,note_range,quant,length_of_chunks,posteriogram,method,augm,annot)

        n_data = len(dataset)
        n_valid = int(0.15*n_data)

        ## Take the first 15% of the dataset as validation files
        ## Shuffle using seed for reproducibility
        random.seed(1234)
        random.shuffle(dataset)
        ## Reset seed
        random.seed()

        dataset_valid = dataset[:n_valid]
        dataset_train = dataset[n_valid:]

        self.valid = dataset_valid
        self.train = dataset_train

        return

    def load_data_test(self,folder,fs,note_range=[0,128],quant=False,posteriogram=False,method='avg',augm=False,annot=None):
        subfolder = os.path.join(folder,'test')
        dataset = self.get_list_of_dataMaps(subfolder,fs,30,note_range,quant,None,posteriogram,method,augm,annot)
        self.test = dataset
        return


    def load_data(self,folder,fs,max_len=None,note_range=[0,128],quant=False,length_of_chunks=None,posteriogram=False,method='avg',norm=False,augm=False,annot_path=None):
        self.note_range = note_range
        if not annot_path is None:
            import cPickle as pickle
            with open(annot_path, 'r') as file:
                annot = pickle.load(file)
            print("Loaded annotation dataset from pickle!")
        else:
            annot = None
        self.load_data_train_valid(folder,fs,max_len,note_range,quant,length_of_chunks,posteriogram,method,augm,annot)
        self.load_data_test(folder,fs,note_range,quant,posteriogram,method,augm,annot)
        self.zero_pad()
        if norm:
            self.normalize_all(folder,quant,method)
        print "Dataset loaded ! "+str(datetime.now())

    def get_n_files(self,subset):
        return len(getattr(self,subset))
    def get_n_notes(self):
        return self.note_range[1]-self.note_range[0]
    def get_len_files(self):
        return self.max_len

    def get_dataset(self,subset,meter=False):
        #Outputs an array containing all the inputs and targets (two 3D-tensor)
        #and the list of the actual lengths of the inputs
        data_list = getattr(self,subset)
        n_files = len(data_list)
        len_file = data_list[0].input.shape[1]
        n_notes = self.get_n_notes()

        if meter:
            inputs = np.zeros([n_files,n_notes+4,len_file])
        else:
            inputs = np.zeros([n_files,n_notes,len_file])
        targets = np.zeros([n_files,n_notes,len_file])
        lengths = np.zeros([n_files])

        i=0
        while i<n_files:
            data = data_list[i]
            input_data = data.input
            target = data.target
            if meter:
                meter_grid = data.meter_grid
                inputs[i,:n_notes,:]=input_data
                inputs[i,n_notes:,:data.length]=meter_grid
            else:
                inputs[i] = input_data
            targets[i] = target
            lengths[i] = data.length
            i += 1

        return inputs, targets, lengths


    def shuffle_one(self,subset):
        data = getattr(self,subset)
        random.shuffle(data)


    def __max_len(self,dataset):
        if dataset == []:
            return 0
        else :
            return max(map(lambda x: x.length, dataset))

    def zero_pad(self):

        max_train = self.__max_len(self.train)
        max_valid = self.__max_len(self.valid)
        max_test = self.__max_len(self.test)
        max_len = max([max_train,max_valid,max_test])
        self.max_len = max_len

        for subset in ["train","valid","test"]:
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

        import cPickle as pickle
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
