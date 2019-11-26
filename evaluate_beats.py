import madmom
import argparse
import pretty_midi as pm
import sounddevice as sd
import os
import numpy as np
import mir_eval


def play_sound(sig, fs=44100):
    # ctrl-C stops the signal and continues the script
    try:
        sd.play(sig, fs)
        status = sd.wait()
    except KeyboardInterrupt:
        sd.stop()
        return

def mix_sounds(sig1,sig2,sig_mix_ratio=0.5):
    len1 = sig1.shape[0]
    len2 = sig2.shape[0]
    if len1 > len2:
        audio = (1-sig_mix_ratio)*sig1 + sig_mix_ratio*np.pad(sig2,(0,len1-len2),'constant')
    else:
        audio = (1-sig_mix_ratio)*np.pad(sig1,(0,len2-len1),'constant') + sig_mix_ratio*sig2
    return audio

def sonify_beats(beats,downbeats=None,subbeats=None):
    midi = pm.PrettyMIDI()

    #Add beats
    if not beats is None:
        bell = pm.Instrument(is_drum=True,program=0)
        for beat in beats:
            note_pm = pm.Note(
                velocity=80, pitch=pm.drum_name_to_note_number('Cowbell'), start=beat, end=beat+0.001)
            bell.notes.append(note_pm)
        midi.instruments.append(bell)

    #Add downbeats
    if not downbeats is None:
        triangle = pm.Instrument(is_drum=True,program=0)
        for downbeat in downbeats:
            note_pm = pm.Note(
                velocity=100, pitch=pm.drum_name_to_note_number('Open Triangle'), start=downbeat, end=downbeat+0.001)
            triangle.notes.append(note_pm)
        midi.instruments.append(triangle)

    #Add subbeats
    if not subbeats is None:
        sidestick = pm.Instrument(is_drum=True,program=0)
        for subbeat in subbeats:
            note_pm = pm.Note(
                velocity=60, pitch=pm.drum_name_to_note_number('Side Stick'), start=subbeat, end=subbeat+0.001)
            sidestick.notes.append(note_pm)
        midi.instruments.append(sidestick)

    audio = midi.fluidsynth()

    return audio

def get_subbeat_divisions(beats,beat_activ):
    """
    Compute the number per beat and locations of sub-beat subdivisions.

    Uses :class:`DBNDownBeatTrackingProcessor` from `madmom` library, as in :func:`get_beats_downbeats_signature`.


    Parameters
    ----------
    beats : 1D numpy array
        positions in seconds of the beats
    beat_activ : 1D numpy array
        beat activations

    Returns
    -------
    int
        Number of subdivisions in each beats (2 or 3)
    1D numpy array
        Positions in seconds of the sub-beat subdivisions (only used for visualisation/debugging)
    """

    n_beats = len(beats)
    n_iter=0
    min_bpm = 110.0
    bpm_incr = 55.0
    for i in range(10):
        proc_beat_track = madmom.features.DBNBeatTrackingProcessor(fps=100,min_bpm=min_bpm,max_bpm=600)
        new_beats = proc_beat_track(beat_activ)
        n_new_beats = len(new_beats)
        if n_new_beats!=n_beats:
            #Different beats found, they correspond to sub-beat level
            if abs(round(n_new_beats/2.0) - n_beats) <= 1:
                #If n_new_beats/2.0 is approx equal to n_beats
                return 2, new_beats
            elif abs(round(n_new_beats/3.0) - n_beats) <= 1:
                #If n_new_beats/3.0 is approx equal to n_beats
                return 3, new_beats
            #Default case; it means that the beat found is not an integer subdivision of the original beats
            #Iterate until we find it


        #If function has not returned, iterate with a higher min_bpm
        min_bpm += bpm_incr
    # Consider that default is binary
    return 2, new_beats



parser = argparse.ArgumentParser()
parser.add_argument('input_folder',type=str)
parser.add_argument('--load',action='store_true',help="use pre-computed values (in the same input folder)")
parser.add_argument('--save',type=str,help="save GT and estimated beats as CSV files in specified folder ('same' saves in input_folder)")
parser.add_argument("--max_len",type=str,help="test on the first max_len seconds of each text file. Anything other than a number will evaluate on whole files. Default is 30s.",
                    default=30)
parser.add_argument('--subbeats',action='store_true',help='also compute subbeat subdivisions')
parser.add_argument('--file',type=str,help="play one specific file only (or any file containing the string)")
parser.add_argument('--play_GT', action='store_true',help="play audio with ground-truth beat and downbeat positions (and subbeats if --subbeat is used)")
parser.add_argument('--play_estim',action='store_true',help="play audio with estimated beat and downbeat positions (and subbeats if --subbeat is used)" )

args = parser.parse_args()

input_folder = args.input_folder


try:
    max_len = float(args.max_len)
    section = [0,max_len]
    print(f"Evaluate on first {args.max_len} seconds")
except:
    max_len = None
    section=None
    print(f"Evaluate on whole files")


all_Fs = []
all_Fs_sub = []

for fn in os.listdir(input_folder):
    if fn.endswith('.wav') and not fn.startswith('.'):
        if args.file is None or args.file in fn:
            filename_input = os.path.join(input_folder,fn)
            print(fn)

            # Get ground truth beats
            midi_data = pm.PrettyMIDI(filename_input.replace('.wav','.mid'))
            beats_GT = midi_data.get_beats()
            beats_GT = beats_GT[beats_GT<max_len]

            downbeats_GT = midi_data.get_downbeats()
            downbeats_GT = downbeats_GT[downbeats_GT<max_len]

            if args.subbeats:
                subbeats_ticks = np.arange(0,midi_data.time_to_tick(max_len),midi_data.resolution/2)
                subbeats_GT = np.array([midi_data.tick_to_time(tick) for tick in subbeats_ticks])


            # Estimate beat positions
            sig_proc = madmom.audio.signal.SignalProcessor(sample_rate=44100, num_channels=1, start=0, stop=max_len,dtype=np.float32)
            sig = sig_proc(filename_input)

            fs = sig.sample_rate
            dur = sig.length

            if args.play_GT:
                sig_mix_ratio = 0.7
                sig_beats = sonify_beats(beats_GT,downbeats_GT,subbeats_GT)
                audio = mix_sounds(sig_beats,sig,sig_mix_ratio)

                play_sound(audio)


            proc_beat = madmom.features.RNNBeatProcessor()
            act_beat = proc_beat(sig)

            proc_beattrack = madmom.features.BeatTrackingProcessor(fps=100)
            beats = proc_beattrack(act_beat)

            F = mir_eval.beat.f_measure(beats_GT,beats)
            all_Fs += [F]
            print(f"Beat F-measure: {F}")
            print(f"GT: {beats_GT}")
            print(f"Est: {beats}")

            if args.subbeats:
                n_subdivisions, subbeats = get_subbeat_divisions(beats,act_beat)
                sub_F = mir_eval.beat.f_measure(subbeats_GT,subbeats)
                all_Fs_sub += [sub_F]
                print(f"Sub-beat F-measure: {sub_F}")
                print(f"Estimated subdivisions: {n_subdivisions}")
                print(f"Time Signatures: {midi_data.time_signature_changes}")
            else:
                subbeats = None

            if args.save is not None:
                if args.save == 'same':
                    save_path = input_folder
                else:
                    save_path = args.save

                np.savetxt(os.path.join(save_path,fn.replace('.wav','_b_gt.csv')),beats_GT)
                np.savetxt(os.path.join(save_path,fn.replace('.wav','_b_est.csv')),beats)

                if args.subbeats:
                    np.savetxt(os.path.join(save_path,fn.replace('.wav','_sb_gt.csv')),subbeats_GT)
                    np.savetxt(os.path.join(save_path,fn.replace('.wav','_sb_est.csv')),subbeats)


            if args.play_estim:
                sig_mix_ratio = 0.7
                sig_beats = sonify_beats(beats,None,subbeats)
                audio = mix_sounds(sig_beats,sig,sig_mix_ratio)

                play_sound(audio)

print(f"Average beat F-measure: {sum(all_Fs)/len(all_Fs)}")
if self.subbeats:
    print(f"Average sub_beat F-measure: {sum(all_Fs_sub)/len(all_Fs_sub)}")
