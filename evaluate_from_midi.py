import os
import shutil
import numpy as np
from dataMaps import DataMaps
import pretty_midi as pm
import pickle
import mir_eval
import argparse
from eval_utils import compute_eval_metrics_frame, filter_short_notes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('estim_path',type=str,help="location of the estimated MIDI")
    parser.add_argument('target_path',type=str,help="location of the ground-truth MIDI")
    parser.add_argument("--max_len",type=str,help="test on the first max_len seconds of each text file. Anything other than a number will evaluate on whole files. Default is 30s.",
                        default=30)

    args = parser.parse_args()

    try:
        max_len = float(args.max_len)
        section = [0,max_len]
        print(f"Evaluate on first {args.max_len} seconds")
    except:
        max_len = None
        section=None
        print(f"Evaluate on whole files")

    # Sampling frequency for the piano roll
    fs=100

    input_folder = args.estim_path
    target_folder = args.target_path

    all_frame = []
    all_on = []
    all_onoff = []

    results = {}

    for midi_name in os.listdir(input_folder):
        if not midi_name.startswith('.') and midi_name.endswith('.mid'):
            print(midi_name)

            input_midi = pm.PrettyMIDI(os.path.join(input_folder,midi_name))
            target_midi = pm.PrettyMIDI(os.path.join(target_folder,midi_name))

            output = (input_midi.get_piano_roll(fs)>0).astype(int)
            output = output[:,int(section[0]*fs):int(section[1]*fs)]
            target = (target_midi.get_piano_roll(fs)>0).astype(int)
            target = target[:,int(section[0]*fs):int(section[1]*fs)]

            P_f,R_f,F_f = compute_eval_metrics_frame(output,target)

            notes_est,intervals_est = [],[]

            for note in sum([instr.notes for instr in input_midi.instruments],[]):
                if section is None or (note.start < section[1] and note.end>section[0]):
                    ### +21-1 because in get_notes_intervals_with_onsets, we add +1 so that pitches are not equal to 0
                    notes_est+= [note.pitch]
                    intervals_est+= [[max(note.start,section[0]),min(note.end,section[1])]]
            notes_est = np.array(notes_est)
            intervals_est = np.array(intervals_est)



            notes_ref,intervals_ref = [],[]

            for note in sum([instr.notes for instr in target_midi.instruments],[]):
                if section is None or (note.start < section[1] and note.end>section[0]):
                    ### +21-1 because in get_notes_intervals_with_onsets, we add +1 so that pitches are not equal to 0
                    notes_ref+= [note.pitch]
                    intervals_ref+= [[max(note.start,section[0]),min(note.end,section[1])]]

            notes_ref = np.array(notes_ref)
            intervals_ref = np.array(intervals_ref)

            # print(len(notes_est))
            notes_est, intervals_est = filter_short_notes(notes_est, intervals_est, 0.05)
            # print(len(notes_est))


            if len(notes_est) == 0:
                P_n_on,R_n_on,F_n_on= 0,0,0
                P_n_onoff,R_n_onoff,F_n_onoff= 0,0,0
            else:
                P_n_on,R_n_on,F_n_on,_ = mir_eval.transcription.precision_recall_f1_overlap(intervals_ref,
                notes_ref,intervals_est,notes_est,pitch_tolerance=0.25,offset_ratio=None,
                onset_tolerance=0.05,offset_min_tolerance=0.05)

                P_n_onoff,R_n_onoff,F_n_onoff,_ = mir_eval.transcription.precision_recall_f1_overlap(intervals_ref,
                notes_ref,intervals_est,notes_est,pitch_tolerance=0.25,offset_ratio=0.2,
                onset_tolerance=0.05,offset_min_tolerance=0.05)

            results[midi_name] = [[P_f,R_f,F_f],[P_n_on,R_n_on,F_n_on],[P_n_onoff,R_n_onoff,F_n_onoff]]

            all_frame += [[P_f,R_f,F_f]]
            all_on += [[P_n_on,R_n_on,F_n_on]]
            all_onoff += [[P_n_onoff,R_n_onoff,F_n_onoff]]
            print(f"Frame P,R,F: {P_f:.3f},{R_f:.3f},{F_f:.3f}, Note P,R,F: {P_n_on:.3f},{R_n_on:.3f},{F_n_on:.3f}, with offsets P,R,F: {P_n_onoff:.3f},{R_n_onoff:.3f},{F_n_onoff:.3f} ")

    pickle.dump(results,open(os.path.join(args.estim_path,'results.p'), "wb"))

    all_frame = np.array(all_frame)
    all_on = np.array(all_on)
    all_onoff = np.array(all_onoff)
    print(all_frame.shape, all_on.shape, all_onoff.shape)
    P_f, R_f, F_f = np.mean(all_frame, axis=0)
    P_n_on,R_n_on,F_n_on = np.mean(all_on, axis=0)
    P_n_onoff,R_n_onoff,F_n_onoff = np.mean(all_onoff, axis=0)
    print(f"Averages: Frame P,R,F: {P_f:.3f},{R_f:.3f},{F_f:.3f}, Note P,R,F: {P_n_on:.3f},{R_n_on:.3f},{F_n_on:.3f}, with offsets P,R,F: {P_n_onoff:.3f},{R_n_onoff:.3f},{F_n_onoff:.3f} ")

    import subprocess
    copy_string = '\t'.join([str(elt) for elt in [P_f, R_f, F_f,P_n_on,R_n_on,F_n_on,P_n_onoff,R_n_onoff,F_n_onoff]])
    subprocess.run("pbcopy", universal_newlines=True, input=copy_string)
