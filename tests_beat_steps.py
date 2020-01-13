import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
import beats_utils


beats_folder = 'results/beat/est_cw'
time_folder = 'results/time/cw_XE'

data_folder = 'data/outputs_default_config_split20p/test'

result_dict_beats = pickle.load(open(os.path.join(beats_folder,'results.p'),'rb'))
result_dict_time = pickle.load(open(os.path.join(time_folder,'results.p'),'rb'))

dict_confidence = {}

for key in result_dict_time.keys():

    # print(key)
    beat_act = np.loadtxt(os.path.join(data_folder,key.replace('.mid','_b_act.csv')))
    beats = np.loadtxt(os.path.join(data_folder,key.replace('.mid','_b_est.csv')))

    spec_flat = beats_utils.get_confidence_spectral_flatness(beat_act)

    beat_std = np.std(beats[1:]-beats[:-1])

    dict_confidence[key] = [spec_flat,beat_std]


######################

f_improv = []
conf_sf = []
conf_std = []

for key in result_dict_time.keys():

    # print(dict_confidence[key][0],result_dict_beats[key][-1][-1], result_dict_time[key][-1][-1])

    f_improv += [result_dict_beats[key][-1][-1] - result_dict_time[key][-1][-1]]

    conf_sf += [dict_confidence[key][0]]
    conf_std += [dict_confidence[key][1]]


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

# plt.subplot(121)
# plt.scatter(conf_sf,f_improv)
# plt.xlabel('Spectral Flatness')
#
# plt.subplot(122)
# plt.scatter(conf_std,f_improv)
# plt.xlabel('Beat STD')
# plt.show()
