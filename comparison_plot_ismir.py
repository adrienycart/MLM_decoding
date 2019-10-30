import dataMaps
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import pretty_midi as pm

norm = Normalize(vmin=0, vmax=1)


note_range=[27,90]
section = [100,300]
# color_map = 'viridis' #Default
color_map = 'inferno'
enhance_colors = True

title_fontsize = 70
labels_fontsize = 40

step = "quant"
lstm_folder = "results/results-20/save/pm-quant-sched"
hmm_folder = "results/results-20/save/hmm-quant"
basename = "MAPS_MUS-grieg_kobold_ENSTDkAm"

lstm_pr = np.load(os.path.join(lstm_folder, basename + "_pr.npy"))
hmm_pr = np.load(os.path.join(hmm_folder, basename + "_pr.npy"))
lstm_prior = np.load(os.path.join(lstm_folder, basename + "_priors.npy"))

data = dataMaps.DataMaps()
data.make_from_file("data/outputs_default_config_split20p/test/" + basename + ".mid", step, section=[0, 30], acoustic_model="kelz")

acoustic = data.input
midi = data.target


midi = midi[note_range[0]-21:note_range[1]-21,section[0]:section[1]]
acoustic = acoustic[note_range[0]-21:note_range[1]-21,section[0]:section[1]]
acoustic_pr = (acoustic>0.5).astype(int)
hmm_pr = hmm_pr[note_range[0]-21:note_range[1]-21,section[0]:section[1]]
lstm_pr = lstm_pr[note_range[0]-21:note_range[1]-21,section[0]:section[1]]
lstm_prior = lstm_prior[note_range[0]-21:note_range[1]-21,section[0]:section[1]]

if enhance_colors:
    acoustic = np.arctanh((acoustic*2-1)*0.85)
    lstm_prior = np.arctanh((lstm_prior*2-1)*0.85)


fig, ax = plt.subplots(6, 1, figsize=(40,50))
fig.suptitle('Comparison for a bad example - 16th note timesteps')

ax[0].imshow(midi, aspect='auto', origin='lower',cmap=color_map)
ax[0].set_title('Target', fontsize=title_fontsize)
ax[1].imshow(acoustic_pr, aspect='auto', origin='lower',cmap=color_map)
ax[1].set_title('Baseline Kelz', fontsize=title_fontsize)
ax[2].imshow(hmm_pr, aspect='auto', origin='lower',cmap=color_map)
ax[2].set_title('HMM', fontsize=title_fontsize)
ax[3].imshow(lstm_pr, aspect='auto', origin='lower',cmap=color_map)
ax[3].set_title('PM+S Output', fontsize=title_fontsize)
ax[4].imshow(acoustic, aspect='auto', origin='lower',cmap=color_map)
ax[4].set_title('Posteriogram', fontsize=title_fontsize)
ax[5].imshow(lstm_prior, aspect='auto', origin='lower',cmap=color_map)
ax[5].set_title('MLM predictions', fontsize=title_fontsize)
for a in ax:
    a.set_xlabel('Timesteps', fontsize=labels_fontsize)
    a.set_ylabel('MIDI Pitch', fontsize=labels_fontsize)
    labels = [pm.note_number_to_name(x) for x in range(*note_range)]
    a.set_yticks([x+0.5 for x in list(range(len(labels)))])
    # a.set_yticklabels(labels,fontsize=5)
    a.set_yticklabels([],fontsize=5)
    a.set_xticklabels([],fontsize=5)
    a.grid(True,axis='y',color='black')

plt.tight_layout(pad=1.5)
# plt.show()
plt.savefig('good_comparison_poster.png')
