from eval_utils import make_midi_from_roll,play_audio,save_midi
from dataMaps import DataMaps, convert_note_to_time

import os
import numpy as np
import argparse


def get_name_from_maps(filename):
    name = filename.split('-')[1:]
    name = '-'.join(name)
    name = name.split('_')[:-1]
    name = '_'.join(name)
    return name


def create_midi_file(gt_midi, model_output_dir, midi_output_dir, filename, step='time', max_len=30):
    """
    Convert csv piano rolls into MIDI files, for a single given MIDI file.
    
    Parameters
    ==========
    gt_midi : string
        The directory containing all of the ground truth MIDI files there are csv files for.
        
    model_output_dir : string
        The directory containing the output csv files to convert.
        
    midi_output_dir : string
        The directory to write the resulting MIDI files to.
        
    filename : string
        The MAPS filename of the MIDI file to convert.
        
    step : string
        The frame step type to use for conversion. Either "time" (default), "quant", or "event".
        
    max_len : int
        The number of seconds of each file to convert. Defaults to 30.
    """
    data = DataMaps()
    data.make_from_file(os.path.join(gt_midi, filename), step, [0, max_len])

    csv_path = os.path.join(model_output_dir, filename.replace('.mid', '_pr.csv'))
    roll = np.loadtxt(csv_path)
    roll_time = convert_note_to_time(roll, data.corresp, max_len=max_len)
    midi_data = make_midi_from_roll(roll_time, 25)

    output_filename = os.path.join(midi_output_dir, get_name_from_maps(filename) + '_' + filename[-6:-4])
    save_midi(midi_data, output_filename + '.mid')

    
    
def create_all_midi_files(gt_midi, model_output_dir, midi_output_dir, step='time', max_len=30):
    """
    Convert csv piano rolls into MIDI files, for each MIDI file in the given directory.
    
    Parameters
    ==========
    gt_midi : string
        The directory containing all of the ground truth MIDI files there are csv files for.
        
    model_output_dir : string
        The directory containing the output csv files to convert.
        
    midi_output_dir : string
        The directory to write the resulting MIDI files to.
        
    step : string
        The frame step type to use for conversion. Either "time" (default), "quant", or "event".
        
    max_len : int
        The number of seconds of each file to convert. Defaults to 30.
    """
    for filename in os.listdir(gt_midi):
        if filename.endswith('.mid') and not filename.startswith('.') and not 'chpn-e01' in filename:
            create_midi_file(gt_midi, model_output_dir, midi_output_dir, filename, step=step, max_len=max_len)

            
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('gt_midi', help="Folder containing the split dataset, or a single GT MIDI file " +
                        "to convert (if --file is given).")
    
    parser.add_argument("--file", help="Given to indicate that gt_midi is only a single file to convert.",
                        action="store_true")
    
    parser.add_argument('-i', '--input', help="Directory containing the model's output csv files which we will" +
                        " convert into MIDI.", required=True)
    parser.add_argument('-o', '--output', help="Directory to write out the generated MIDI files.", required=True)
    
    parser.add_argument("--step", type=str, choices=["time", "quant", "event"], help="Change the step type " +
                        "for frame timing. Either time (default), quant (for 16th notes), or event (for onsets).",
                        default="time")
    
    parser.add_argument("--max_len", type=str, help="test on the first max_len seconds of each text file. " +
                        "Anything other than a number will evaluate on whole files. Default is 30s.",
                        default=30)
    
    args = parser.parse_args()
    
    try:
        max_len = float(args.max_len)
    except:
        max_len = None
        
    if args.file:
        create_midi_file(os.path.dirname(args.gt_midi), args.input, args.output, os.path.basename(args.gt_midi),
                              step=args.step, max_len=max_len)
    else:
        create_all_midi_files(args.gt_midi, args.input, args.output, step=args.step, max_len=max_len)