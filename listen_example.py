from eval_utils import make_midi_from_roll,play_audio,save_midi
from dataMaps import DataMaps, convert_note_to_time
from mlm_training.utils import safe_mkdir

import os
import numpy as np


def get_name_from_maps(filename):
    name = filename.split('-')[1:]
    name = '-'.join(name)
    name = name.split('_')[:-1]
    name = '_'.join(name)
    return name




data_folders = {}
data_folders[''] = "results/quant/save-quant"
data_folders['gtweight'] = "results/save-gtweight/save-quant-gtweight"
data_folders['old'] = "results/quant/autoweight_k40_b100_h20"
data_folders['baseline'] = "results/baseline/quant"
data_folders['hmm'] = "results/save-hmm/save-quant-hmm"
GT_folder = "data/outputs_default_config_split/test"

# filename  = "MAPS_MUS-chpn_op35_1_ENSTDkAm.mid"


output = 'results/midi_outputs/'


###### ONE FILE ONLY
# for filename in [filename]:
######


###### WHOLE FOLDER
for filename in os.listdir(GT_folder):
######

    if filename.endswith('.mid') and not filename.startswith('.') and not 'chpn-e01' in filename:

        example_name = get_name_from_maps(filename)+'_'+filename[-6:-4]
        example_folder = os.path.join(output,example_name)
        safe_mkdir(example_folder)


        output_filename = os.path.join(example_folder,example_name)

        data = DataMaps()
        data.make_from_file(os.path.join(GT_folder,filename),'quant',[0,30])

        for suffix, folder in data_folders.items():
            csv_path = os.path.join(folder,filename.replace('.mid','_pr.csv'))
            roll = np.loadtxt(csv_path)
            roll_time = convert_note_to_time(roll,data.corresp,max_len=30)
            midi_data = make_midi_from_roll(roll_time,25)
            save_midi(midi_data,output_filename+"_"+suffix+'.mid')


        data_GT = DataMaps()
        data_GT.make_from_file(os.path.join(GT_folder,filename),'time',[0,30])
        midi_data_GT = make_midi_from_roll(data_GT.target,25)
        save_midi(midi_data_GT,output_filename+'_GT.mid')
