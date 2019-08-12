import dataMaps
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import pretty_midi as pm

norm = Normalize(vmin=0, vmax=1)

step = "quant"
lstm_folder = "results/results-20/save/pm-quant-sched"
hmm_folder = "results/results-20/save/hmm-quant"
basename = "MAPS_MUS-grieg_butterfly_ENSTDkCl"

lstm_pr = np.load(os.path.join(lstm_folder, basename + "_pr.npy"))
hmm_pr = np.load(os.path.join(hmm_folder, basename + "_pr.npy"))
lstm_prior = np.load(os.path.join(lstm_folder, basename + "_priors.npy"))

data = dataMaps.DataMaps()
data.make_from_file("data/outputs_default_config_split20p/test/" + basename + ".mid", step, section=[0, 30], acoustic_model="kelz")

acoustic = data.input
midi = data.target

fig, ax = plt.subplots(6, 1, figsize=(50,50))
fig.suptitle('Comparison for a bad example - 16th note timesteps')

ax[0].imshow(midi, aspect='auto', origin='lower')
ax[0].set_title('Target', fontsize=30)
ax[1].imshow((acoustic>0.5).astype(int), aspect='auto', origin='lower')
ax[1].set_title('Baseline Kelz', fontsize=30)
ax[2].imshow(hmm_pr, aspect='auto', origin='lower')
ax[2].set_title('HMM', fontsize=30)
ax[3].imshow(lstm_pr, aspect='auto', origin='lower')
ax[3].set_title('PM+S Output', fontsize=30)
ax[4].imshow(acoustic, aspect='auto', origin='lower')
ax[4].set_title('Posteriogram', fontsize=30)
ax[5].imshow(lstm_prior, aspect='auto', origin='lower')
ax[5].set_title('MLM predictions', fontsize=30)
for a in ax:
    a.set_xlabel('Timesteps', fontsize=20)
    a.set_ylabel('MIDI Pitch', fontsize=20)
    labels = [pm.note_number_to_name(x) for x in range(21,109)]
    a.set_yticks([x+0.5 for x in list(range(len(labels)))])
    a.set_yticklabels(labels,fontsize=5)
    a.grid(True,axis='y',color='black')

# plt.tight_layout()
# plt.show()
plt.savefig('bad_comparison.png')
