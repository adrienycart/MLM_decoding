import matplotlib
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
import beats_utils
import mir_eval
import pretty_midi as pm

def normalize(data):
    return (data-np.min(data))/(np.max(data)-np.min(data))

# beats_folder = 'results/beat/est_cw'
# time_folder = 'results/time/cw_XE'

beats_folder = 'results/beat/est_prior'
time_folder = 'results/beat/gt_prior'

data_folder = 'data/outputs_default_config_split20p/test'

result_dict_beats = pickle.load(open(os.path.join(beats_folder,'results.p'),'rb'))
result_dict_time = pickle.load(open(os.path.join(time_folder,'results.p'),'rb'))

dict_confidence = {}

for key in result_dict_time.keys():

    # print(key)
    beat_act = np.loadtxt(os.path.join(data_folder,key.replace('.mid','_b_act.csv')))
    beats = np.loadtxt(os.path.join(data_folder,key.replace('.mid','_b_est.csv')))
    beats_GT = np.loadtxt(os.path.join(data_folder,key.replace('.mid','_b_GT.csv')))

    spec_flat = beats_utils.get_confidence_spectral_flatness(beat_act)

    beat_std = np.std(beats[1:]-beats[:-1])

    beat_F1 = mir_eval.beat.f_measure(beats_GT,beats)

    avg_beat_est = np.mean(beats[1:]-beats[:-1])
    avg_beat_gt = np.mean(beats_GT[1:]-beats_GT[:-1])

    midi_data = pm.PrettyMIDI(os.path.join(data_folder,key))
    all_notes = sum([instr.notes for instr in midi_data.instruments],[])
    all_notes = [note for note in all_notes if note.start < 30]
    avg_note_duration = np.mean([note.end-note.start for note in all_notes])

    dict_confidence[key] = [spec_flat,beat_std,beat_F1,avg_beat_est,avg_beat_gt,avg_note_duration]


######################

f_improv = []
conf_sf = []
conf_std = []
conf_f1 = []
avg_beat_ratio = []
avg_note_durations = []

for key in result_dict_time.keys():

    # print(dict_confidence[key][0],result_dict_beats[key][-1][-1], result_dict_time[key][-1][-1])

    f_improv += [result_dict_beats[key][-1][-1] - result_dict_time[key][-1][-1]]

    conf_sf += [dict_confidence[key][0]]
    conf_std += [dict_confidence[key][1]]
    conf_f1 += [dict_confidence[key][2]]
    avg_beat_ratio += [dict_confidence[key][4]/dict_confidence[key][3]]
    avg_note_durations += [dict_confidence[key][5]]

cmap = matplotlib.cm.get_cmap('inferno')
avg_note_durations = np.array(avg_note_durations)
avg_note_durations_log = np.log(avg_note_durations)
avg_note_durations_norm = normalize(avg_note_durations_log)
colors = [cmap(val) for val in avg_note_durations_norm]



range_sf = np.arange(0,1,0.05)
range_std = np.arange(0,0.5,0.05)

results = np.zeros([len(range_sf),len(range_std)])

for i,thresh_sf in enumerate(range_sf):
    for j,thresh_std  in enumerate(range_std):

        decision = []

        for key in result_dict_time.keys():

            if dict_confidence[key][0] > thresh_sf and dict_confidence[key][1] > thresh_std:
                decision += [result_dict_time[key][-1][-1]]
            else:
                decision += [result_dict_beats[key][-1][-1]]

        results[i,j] = np.mean(decision)

best_config = np.argmax(results)
best_config = np.unravel_index(best_config, results.shape)
best_thresh_sf = range_sf[best_config[0]]
best_thresh_std = range_std[best_config[1]]
best = results[best_config]

print(f"Best: {best}, thresh_sf:{best_thresh_sf}, thresh_std: {best_thresh_std}")
print(f"(time only: {np.mean([result_dict_time[key][-1][-1] for key in result_dict_time])})")
print(f"(beat only: {np.mean([result_dict_beats[key][-1][-1] for key in result_dict_time])})")

# plt.subplot(131)
# plt.scatter(conf_sf,f_improv)
# plt.xlabel('Spectral Flatness')
#
#
# plt.subplot(132)
# plt.scatter(conf_std,f_improv)
# plt.xlabel('Beat STD')
#
# plt.subplot(133)
# plt.scatter(conf_f1,f_improv)
# plt.xlabel('Beat F1')
#
# plt.suptitle('Improvement in notewise-F1 from 40ms to beat steps\nvs. beat tracking confidence metrics ')
#
# plt.show()



# print("Improvement for each F1 bin")
# conf_f1 = np.array(conf_f1)
# f_improv = np.array(f_improv)
# step=0.1
# improv_means = []
# improv_std = []
# for i in np.arange(0,1,step):
#     data = f_improv[np.logical_and(i<=conf_f1,conf_f1<=i+step)]
#     mean = np.mean(data)
#     std = np.std(data)
#     print(f"{i},{i+step}: {mean}")
#     improv_means += [mean]
#     improv_std += [std]
#
# plt.errorbar(np.arange(10)+0.5,improv_means,yerr=improv_std,capsize=1,ecolor='black',elinewidth=0.5)
# plt.xticks(range(11),np.round(np.arange(0,1.1,0.1),1))
# plt.grid(axis='x')
# plt.scatter(conf_f1*10,f_improv,alpha=0.7,edgecolors='black',linewidths=0.5)
# plt.xlabel("Beat-tracking F1")
# plt.ylabel("Notewise F1 improvement from GT to EST beat steps")
# plt.title("Improvement in notewise F1 \nfrom PM+GT beat steps to PM+EST beat steps")
# plt.show()


# print("Improvement for each average beat ratio bin")
# avg_beat_ratio = np.array(avg_beat_ratio)
# f_improv = np.array(f_improv)
# bin_centers = [0.5,0.66,1,2,3]
# bin_edges = [bin_centers[i]+(bin_centers[i+1]-bin_centers[i])/2.0 for i in range(len(bin_centers)-1)]
# bin_edges = [0]+bin_edges+[4]
#
# improv_means = []
# improv_std = []
# for i in range(len(bin_edges)-1):
#     data = f_improv[np.logical_and(bin_edges[i]<=avg_beat_ratio,avg_beat_ratio<=bin_edges[i+1])]
#     mean = np.mean(data)
#     std = np.std(data)
#     print(f"{bin_edges[i]},{bin_edges[i+1]}: {mean}")
#     improv_means += [mean]
#     improv_std += [std]
#
# plt.errorbar(bin_centers,improv_means,yerr=improv_std,capsize=1,ecolor='black',elinewidth=0.5)
# plt.xticks(bin_edges,np.round(bin_edges,2))
# plt.grid(axis='x')
# plt.scatter(avg_beat_ratio,f_improv,c=colors,alpha=0.7,edgecolors='black',linewidths=0.5)
# plt.xlabel("Average beat duration ratio (GT / EST)")
# plt.ylabel("Notewise F1 improvement from GT to EST beat steps")
# plt.title("Improvement in notewise F1 \nfrom PM+GT beat steps to PM+EST beat steps")
# plt.show()
# plt.scatter(avg_beat_ratio,f_improv,alpha=0.7,edgecolors='black',linewidths=0.5)
# plt.show()

print("Improvement for each average note duration bin")
f_improv = np.array(f_improv)
bin_edges = [0,0.25,0.5,0.75,1,1.5,2]
bin_centers = [bin_edges[i]+(bin_edges[i+1]-bin_edges[i])/2 for i in range(len(bin_edges)-1)]
improv_means = []
improv_std = []
for i in range(len(bin_edges)-1):
    data = f_improv[np.logical_and(bin_edges[i]<= avg_note_durations ,avg_note_durations<=bin_edges[i+1])]
    mean = np.mean(data)
    std = np.std(data)
    print(f"{bin_edges[i]},{bin_edges[i+1]}: {mean}")
    improv_means += [mean]
    improv_std += [std]

colors = [cmap(val) for val in conf_f1]
plt.errorbar(bin_centers,improv_means,yerr=improv_std,capsize=1,ecolor='black',elinewidth=0.5)
plt.xticks(bin_edges,np.round(bin_edges,2))
plt.grid(axis='x')
plt.scatter(avg_note_durations,f_improv,c=colors,alpha=0.7,edgecolors='black',linewidths=0.5)
plt.xlabel("Average note duration")
plt.ylabel("Notewise F1 improvement from GT to EST beat steps")
plt.title("Improvement in notewise F1 \nfrom PM+GT beat steps to PM+EST beat steps")
plt.show()
plt.scatter(avg_beat_ratio,f_improv,alpha=0.7,edgecolors='black',linewidths=0.5)
plt.show()
