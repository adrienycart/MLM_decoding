import pretty_midi as pm
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
from dataMaps import DataMaps, DataMapsBeats, convert_note_to_time

plt.rcParams.update({'font.family' : 'serif'})

fs=25

filename = 'MAPS_MUS-pathetique_3_ENSTDkAm.mid'
save_dest = 'plots_thesis/good_longer.pdf'
note_min = 40
note_max = 90
show_piano = 5
min_step = 0
min_len = 0
max_len = 10
max_step = max_len*fs


# filename = 'MAPS_MUS-liz_et_trans5_ENSTDkCl.mid'
# save_dest = 'plots_thesis/good_shorter.pdf'
# note_min = 50
# note_max = 108
# show_piano = 5
# min_step = 0
# min_len = 0
# max_len = 10
# max_step = max_len*fs

# filename = 'MAPS_MUS-pathetique_2_ENSTDkAm.mid'
# save_dest = 'plots_thesis/bad_shorter.pdf'
# note_min = 40
# note_max = 75
# show_piano = 5
# min_step = 0
# min_len = 0
# max_len = 10
# max_step = max_len*fs

# filename = 'MAPS_MUS-mz_311_1_ENSTDkCl.mid'
# save_dest = 'plots_thesis/bad_longer.pdf'
# note_min = 31
# note_max = 95
# show_piano = 5
# min_step = 0
# min_len = 0
# max_len = 10
# max_step = max_len*fs




folder = "data/outputs_default_config_split20p/test/"
folder_est = 'results/beat/baseline_est'
folder_gt = 'results/beat/baseline_gt'










filename = os.path.join(folder,filename)

data = DataMaps()
data.make_from_file(filename,'time',section=[0,30], acoustic_model='kelz')
roll_gt = data.target

data_est = DataMapsBeats()
data_est.make_from_file(filename,False,section=[0,30], acoustic_model='kelz')
est_roll = np.loadtxt(os.path.join(folder_est,os.path.basename(filename).replace('.mid','_pr.csv')))
# est_roll = (data_est.input>0.5).astype(float)
roll_time_est = convert_note_to_time(est_roll,data_est.corresp,25,30)

data_gt = DataMapsBeats()
data_gt.make_from_file(filename,True,section=[0,30], acoustic_model='kelz')
gt_roll = np.loadtxt(os.path.join(folder_gt,os.path.basename(filename).replace('.mid','_pr.csv')))
# gt_roll = (data_gt.input>0.5).astype(float)
roll_time_gt = convert_note_to_time(gt_roll,data_gt.corresp,25,30)

print(est_roll.shape)
print(gt_roll.shape)

# plt.subplot(221)
# plt.imshow(data_est.input,aspect='auto',origin='lower')
# plt.subplot(222)
# plt.imshow(est_roll,aspect='auto',origin='lower')
#
# plt.subplot(223)
# plt.imshow(data_gt.input,aspect='auto',origin='lower')
# plt.subplot(224)
# plt.imshow(gt_roll,aspect='auto',origin='lower')
# plt.show()

midi_data = pm.PrettyMIDI(filename)


# plt.subplot(311)
# plt.imshow(roll_time_est,aspect='auto',origin='lower')
# plt.subplot(312)
# plt.imshow(roll_time_gt,aspect='auto',origin='lower')
# plt.subplot(313)
# plt.imshow((midi_data.get_piano_roll(25)[21:109,:30*25]>0).astype(int),aspect='auto',origin='lower')
# plt.show()


GT_beats = np.loadtxt(filename.replace('.mid','_b_gt.csv'))
EST_beats = np.loadtxt(filename.replace('.mid','_b_est.csv'))



GT_beats = GT_beats[GT_beats<max_len]
EST_beats = EST_beats[EST_beats<max_len]

roll_gt = roll_gt[note_min-21:note_max-21,min_step:max_step]
roll_time_est = roll_time_est[note_min-21:note_max-21,min_step:max_step]
roll_time_gt = roll_time_gt[note_min-21:note_max-21,min_step:max_step]

labels = list(range(note_min,note_max))
labels = [pm.note_number_to_name(x) for x in labels]
# labels = [ label if 'C' in label and not '#' in label else '' for label in labels]
scale = ['#' not in label for label in labels]
n_labels = len(labels)

my_cm = matplotlib.cm.get_cmap(matplotlib.rcParams['image.cmap'])

mapped_data_gt = my_cm((roll_gt>0).astype(float))
mapped_data_beat_est = my_cm((roll_time_est>0).astype(float))
mapped_data_beat_gt = my_cm((roll_time_gt>0).astype(float))

piano = np.full([mapped_data_gt.shape[0],show_piano,4],0,dtype=float)
piano[:,:,-1]=1
piano[scale,:,:] = [1,1,1,1]

print piano.shape, mapped_data_beat_est.shape

mapped_data_gt = np.concatenate((piano,mapped_data_gt),axis=1)
mapped_data_beat_est = np.concatenate((piano,mapped_data_beat_est),axis=1)
mapped_data_beat_gt = np.concatenate((piano,mapped_data_beat_gt),axis=1)

fig, axes = plt.subplots(3,1,figsize=[7,10])
## ground truth
axes[0].imshow(mapped_data_gt,aspect='auto',origin='lower',extent=[-show_piano,roll_gt.shape[1],-0.5,roll_gt.shape[0]-0.5])
## Beat GT
axes[1].imshow(mapped_data_beat_gt,aspect='auto',origin='lower',extent=[-show_piano,roll_time_gt.shape[1],-0.5,roll_time_gt.shape[0]-0.5])
for beat in GT_beats:
    axes[1].plot([round(beat*fs),round(beat*fs)],[-0.5,roll_time_gt.shape[0]-0.5],color='white',linewidth=0.4)
# for beat in data_gt.corresp[data_gt.corresp<max_len]:
#     axes[1].plot([round(beat*fs),round(beat*fs)],[-0.5,roll_time_gt.shape[0]-0.5],color='grey',linewidth=0.2)
## Beat EST
axes[2].imshow(mapped_data_beat_est,aspect='auto',origin='lower',extent=[-show_piano,roll_time_est.shape[1],-0.5,roll_time_est.shape[0]-0.5])
for beat in EST_beats:
    axes[2].plot([round(beat*fs),round(beat*fs)],[-0.5,roll_time_est.shape[0]-0.5],color='white',linewidth=0.4)
# for beat in data_est.corresp[data_est.corresp<max_len]:
#     axes[2].plot([round(beat*fs),round(beat*fs)],[-0.5,roll_time_est.shape[0]-0.5],color='grey',linewidth=0.2)

for ax in axes:
    ax.set_yticks([x+0.5 for x in list(range(n_labels))])
    ax.set_yticklabels(labels,fontsize=8)
    ax.tick_params(labelleft='off')
    ax.set_ylabel('MIDI Pitch',fontsize=10)
    ax.grid(True,axis='y',color='black')
    ax.set_xlabel('Timesteps',fontsize=10)
plt.tight_layout()
plt.savefig(save_dest)
# plt.show()
