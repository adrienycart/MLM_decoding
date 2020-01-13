import madmom
import argparse
import pretty_midi as pm
try:
    import sounddevice as sd
    SOUND = True
except OSError:
    print('Sound not available! All "play" options are disabled')
    SOUND = False
import os
import warnings
import numpy as np
import mir_eval
from scipy import stats
import eval_utils
import beats_utils


parser = argparse.ArgumentParser()
parser.add_argument('input_folder',type=str)
parser.add_argument('--load',action='store_true',help="use pre-computed values (in the same input folder)")
parser.add_argument('--save',type=str,help="save GT and estimated beats as CSV files in specified folder ('same' saves in input_folder)")
parser.add_argument("--max_len",type=str,help="test on the first max_len seconds of each text file. Anything other than a number will evaluate on whole files. Default is 30s.",
                    default=30)
parser.add_argument('--midi',action='store_true',help="synthesize MIDI file instead of loading wav files")
parser.add_argument('--subbeats',action='store_true',help='also compute subbeat subdivisions')
parser.add_argument('--file',type=str,help="analyse one specific file only (or any file containing the string)")
parser.add_argument('--play_GT', action='store_true',help="play audio with ground-truth beat and downbeat positions (and subbeats if --subbeat is used)")
parser.add_argument('--play_estim',action='store_true',help="play audio with estimated beat and downbeat positions (and subbeats if --subbeat is used)" )
parser.add_argument('--play_both',action='store_true',help="play audio with both ground-truth and estimated beat and downbeat positions (and subbeats if --subbeat is used)" )

args = parser.parse_args()

input_folder = args.input_folder

if not SOUND and (args.play_GT or args.play_estim or args.play_both):
    warnings.warn("No sound can be played on this device! Ignoring --play options")


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

if args.midi:
    extension = '.mid'
else:
    extension = '.wav'


for fn in os.listdir(input_folder):
    if fn.endswith(extension) and not fn.startswith('.'):
        if args.file is None or args.file in fn:
            filename_input = os.path.join(input_folder,fn)
            print(fn)

            if args.midi:
                midi_name = filename_input
            else:
                midi_name = filename_input.replace('.wav','.mid')

            # Get ground truth beats
            midi_data = pm.PrettyMIDI(midi_name)
            beats_GT = midi_data.get_beats()
            if max_len is not None:
                beats_GT = beats_GT[beats_GT<max_len]

            downbeats_GT = midi_data.get_downbeats()
            if max_len is not None:
                downbeats_GT = downbeats_GT[downbeats_GT<max_len]

            if args.subbeats:
                subbeats_ticks = np.arange(0,midi_data.time_to_tick(max_len),midi_data.resolution/2)
                subbeats_GT = np.array([midi_data.tick_to_time(tick) for tick in subbeats_ticks])
            else:
                subbeats_GT = None

            # Estimate beat positions
            if args.midi:
                sig = midi_data.fluidsynth()
                # print(midi_data.instruments)
                if max_len is not None:
                    sig = sig[:int(max_len*44100)]
            else:
                sig_proc = madmom.audio.signal.SignalProcessor(sample_rate=44100, num_channels=1, start=0, stop=max_len,dtype=np.float32)
                sig = sig_proc(filename_input)

                fs = sig.sample_rate
                dur = sig.length


            if args.load:
                beats = np.loadtxt(filename_input.replace(extension,'_b_est.csv'))
                if max_len is not None:
                    beats = beats[beats<max_len]

            else:
                proc_beat = madmom.features.RNNBeatProcessor()
                act_beat = proc_beat(sig)

                proc_onsets = madmom.features.SpectralOnsetProcessor(method='superflux')
                act_onsets = proc_onsets(sig)

                confidence1,spec_norm1 = beats_utils.get_confidence_entropy(act_onsets)
                confidence2,spec_norm2 = beats_utils.get_confidence_entropy(act_beat)
                confidence3 = beats_utils.get_confidence_spectral_flatness(act_beat)
                confidence4 = beats_utils.get_confidence_spectral_flatness(act_onsets)
                print(np.mean(confidence1),np.mean(confidence2))
                print(np.mean(confidence3),np.mean(confidence4))
                # import matplotlib.pyplot as plt
                # plt.subplot(221)
                # plt.plot(act_beat)
                # plt.subplot(222)
                # plt.plot(act_onsets)
                # plt.subplot(223)
                # plt.imshow(spec_norm1,aspect='auto',origin='lower')
                # plt.subplot(224)
                # plt.imshow(spec_norm1,aspect='auto',origin='lower')
                # plt.show()

                proc_beattrack = madmom.features.BeatTrackingProcessor(fps=100)
                beats = proc_beattrack(act_beat)

            F = mir_eval.beat.f_measure(beats_GT,beats)
            all_Fs += [F]
            print(f"Beat F-measure: {F}")
            # print(f"GT: {beats_GT}")
            # print(f"Est: {beats}")

            if args.subbeats:
                if args.load:
                    subbeats = np.loadtxt(filename_input.replace(extension,'_sb_est.csv'))
                else:
                    n_subdivisions, subbeats = beats_utils.get_subbeat_divisions(beats,act_beat)
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

                np.savetxt(os.path.join(save_path,fn.replace(extension,'_b_gt.csv')),beats_GT)
                np.savetxt(os.path.join(save_path,fn.replace(extension,'_b_est.csv')),beats)

                if args.subbeats:
                    np.savetxt(os.path.join(save_path,fn.replace(extension,'_sb_gt.csv')),subbeats_GT)
                    np.savetxt(os.path.join(save_path,fn.replace(extension,'_sb_est.csv')),subbeats)

            if (args.play_GT or args.play_both) and SOUND:
                sig_mix_ratio = 0.7
                sig_beats = beats_utils.sonify_beats(beats_GT,None,subbeats_GT)
                if max_len is not None:
                    sig_beats = sig_beats[:int(max_len*44100)]
                audio = eval_utils.mix_sounds(sig_beats,sig,sig_mix_ratio)
                eval_utils.play_audio(audio)

            if (args.play_estim  or args.play_both)  and SOUND:
                sig_mix_ratio = 0.7
                sig_beats = beats_utils.sonify_beats(beats,None,subbeats)
                if max_len is not None:
                    sig_beats = sig_beats[:int(max_len*44100)]
                audio = eval_utils.mix_sounds(sig_beats,sig,sig_mix_ratio)
                eval_utils.play_audio(audio)

print(f"Average beat F-measure: {sum(all_Fs)/len(all_Fs)}")
if args.subbeats:
    print(f"Average sub_beat F-measure: {sum(all_Fs_sub)/len(all_Fs_sub)}")
