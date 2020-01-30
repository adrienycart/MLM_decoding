import os
import shutil
import numpy as np
from dataMaps import DataMaps
import pretty_midi as pm
import pickle
import mir_eval
from eval_utils import compute_eval_metrics_frame


section = [0,30]

fs=100

input_folder = 'data/midi_adsr'
target_folder = "data/outputs_default_config_split20p/test"

all_frame = []
all_on = []
all_onoff = []



for midi_name in os.listdir(input_folder):
    if not midi_name.startswith('.') and midi_name.endswith('.mid') and not 'chpn-e01' in midi_name:
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
                notes_est+= [note.pitch-20]
                intervals_est+= [[max(note.start,section[0]),min(note.end,section[1])]]
        notes_est = np.array(notes_est)
        intervals_est = np.array(intervals_est)



        notes_ref,intervals_ref = [],[]

        for note in sum([instr.notes for instr in target_midi.instruments],[]):
            if section is None or (note.start < section[1] and note.end>section[0]):
                ### +21-1 because in get_notes_intervals_with_onsets, we add +1 so that pitches are not equal to 0
                notes_ref+= [note.pitch-20]
                intervals_ref+= [[max(note.start,section[0]),min(note.end,section[1])]]

        notes_ref = np.array(notes_ref)
        intervals_ref = np.array(intervals_ref)


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

        all_frame += [[P_f,R_f,F_f]]
        all_on = [[P_n_on,R_n_on,F_n_on]]
        all_onoff = [[P_n_onoff,R_n_onoff,F_n_onoff]]
        print(f"Frame P,R,F: {P_f:.3f},{R_f:.3f},{F_f:.3f}, Note P,R,F: {P_n_on:.3f},{R_n_on:.3f},{F_n_on:.3f}, with offsets P,R,F: {P_n_onoff:.3f},{R_n_onoff:.3f},{F_n_onoff:.3f} ")


P_f, R_f, F_f = np.mean(all_frame, axis=0)
P_n_on,R_n_on,F_n_on = np.mean(all_on, axis=0)
P_n_onoff,R_n_onoff,F_n_onoff = np.mean(all_onoff, axis=0)
print(f"Averages: Frame P,R,F: {P_f:.3f},{R_f:.3f},{F_f:.3f}, Note P,R,F: {P_n_on:.3f},{R_n_on:.3f},{F_n_on:.3f}, with offsets P,R,F: {P_n_onoff:.3f},{R_n_onoff:.3f},{F_n_onoff:.3f} ")
