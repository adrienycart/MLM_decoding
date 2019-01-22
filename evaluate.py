from dataset import DatasetMaps
from mlm_training.model import Model, make_model_param
from decode import decode

import os

import tensorflow as tf
import pretty_midi as pm
import numpy as np
from datetime import datetime
import sys


parser = argparse.ArgumentParser()
parser.add_argument('save_path',type=str,help="folder to the checkpoint to load (inside ckpt folder)")
parser.add_argument('data_path',type=str,help="folder containing the split dataset")
parser.add_argument('-quant',action='store_true',help="use quantised timesteps")
parser.add_argument('-save',type=str,help="location to save the computed results")

args = parser.parse_args()

if args.quant:
    fs = 4
else:
    fs = 100

#Only evaluateon first 30 seconds of each file (as usually done)
max_len = 30

note_range = [21,109]
note_min = note_range[0]
note_max = note_range[1]

n_hidden = 256


# data = Dataset()
# data.load_data('data/Piano-midi.de/',note_range=note_range,
#     fs=fs,max_len=max_len,quant=args.quant)

data = Dataset()
data.load_data("data/test_dataset/",note_range=note_range,
    fs=fs,max_len=max_len,quant=args.quant)


results = {}

model_param = make_model_param()
model_param['n_hidden']=n_hidden
model_param['n_steps']=1
model_param['n_notes']=88

model = Model(model_param)

for pr in data.test:
    "TODO"

# result = get_best_eval_metrics(data,model,save_path,verbose=True)


results[n_hidden][learning_rate] = result


# import cPickle as pickle
# pickle.dump(results, open(os.path.join("ckpt",base_path,'results_4.p'), "wb"))

#print "Computation end : "+str(datetime.now())
