from eval_utils import make_midi_from_roll,play_audio,save_midi
from dataMaps import DataMaps, convert_note_to_time

import os
import numpy as np


def get_name_from_maps(filename):
    name = filename.split('-')[1:]
    name = '-'.join(name)
    name = name.split('_')[:-1]
    name = '_'.join(name)
    return name



result_folder = "results/quant/autoweight_k40_b100_h20"
baseline_folder = "results/baseline/quant"
input_folder = "data/outputs_default_config_split/test"
filename  = "MAPS_MUS-chpn-p14_ENSTDkAm.mid"

output = 'results/midi_outputs/'


###### ONE FILE ONLY
# for filename in [filename]:
######


###### WHOLE FOLDER
for filename in os.listdir(input_folder):
######

    if filename.endswith('.mid') and not filename.startswith('.') and not 'chpn-e01' in filename:
        output_filename = os.path.join(output,get_name_from_maps(filename)+'_'+filename[-6:-4])

        data = DataMaps()
        data.make_from_file(os.path.join(input_folder,filename),'quant',[0,30])

        csv_path = os.path.join(result_folder,filename.replace('.mid','_pr.csv'))
        roll = np.loadtxt(csv_path)
        roll_time = convert_note_to_time(roll,data.corresp,max_len=30)
        midi_data = make_midi_from_roll(roll_time,25)


        csv_path_baseline = os.path.join(baseline_folder,filename.replace('.mid','_pr.csv'))
        roll_baseline = np.loadtxt(csv_path_baseline)
        roll_time_baseline = convert_note_to_time(roll_baseline,data.corresp,max_len=30)
        midi_data_baseline = make_midi_from_roll(roll_time_baseline,25)


        save_midi(midi_data,output_filename+'.mid')
        save_midi(midi_data_baseline,output_filename+'_baseline.mid')
