from mlm_training.dataset import Dataset, ground_truth, safe_mkdir
from mlm_training.model import Model, make_model_from_dataset, make_save_path, make_model_param, make_train_param

import os

import tensorflow as tf
import pretty_midi as pm
import numpy as np
from datetime import datetime
import sys


note_range = [21,109]
note_min = note_range[0]
note_max = note_range[1]
fs=4
max_len = 60 #3 seconds files only

n_hiddens = [ 128] #number of features in hidden layer
learning_rates = [1, 0.01]


train_param = make_train_param()
train_param['epochs']=100
train_param['batch_size']=10
train_param['display_per_epoch']=5
train_param['save_step']=1
train_param['max_to_keep']=1
train_param['summerize']=True

print "Computation start : "+str(datetime.now())

data = Dataset()
data.load_data('data/test_dataset/',note_range=note_range,
    fs=fs,max_len=max_len,quant=True)
data.transpose_all()

base_path = 'test'

for n_hidden in n_hiddens:
    for learning_rate in learning_rates:

        model_param = make_model_param()
        model_param['n_hidden']=n_hidden
        model_param['learning_rate']=learning_rate


        save_path = make_save_path(base_path,model_param)
        log_path = os.path.join("ckpt",save_path)
        safe_mkdir(log_path)
        # f= open(os.path.join(log_path,"log.txt"), 'w')
        # sys.stdout = f

        print "________________________________________"
        print "Hidden nodes = "+str(n_hidden)+", Learning rate = "+str(learning_rate)
        print "________________________________________"
        print "."

        model = make_model_from_dataset(data,model_param)
        model.train(data,save_path=save_path,train_param=train_param)
        tf.reset_default_graph()
        print "."
        print "."
        print "."

#print "Computation end : "+str(datetime.now())
