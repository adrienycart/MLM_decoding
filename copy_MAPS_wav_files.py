import os
import shutil

data = 'data/outputs_default_config_split20p/'
maps = '/import/c4dm-datasets/Data4Sid/MAPS/MAPS'

for subfolder in ['train','valid','test']:
    subfolder = os.path.join(data,subfolder)
    for fn in os.listdir(subfolder):
        if fn.endswith('.mid') and not fn.startswith('.'):
            split_str = fn.split('_')
            piano_name = split_str[-1].split('.')[0]

            print(fn)
            src = os.path.join(maps,piano_name,'MUS',fn.replace('.mid','.wav'))
            shutil.copy2(src,subfolder)
