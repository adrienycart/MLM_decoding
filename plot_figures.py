import matplotlib.pyplot as plt
import numpy as np
import pickle
import os



folder_1 = 'results/quant/autoweight_k40_b100_h20'
folder_2 = 'results/baseline/quant'



results_1 = pickle.load(open(os.path.join(folder_1,'results.p'), "rb"))
results_2 = pickle.load(open(os.path.join(folder_2,'results.p'), "rb"))

keys = list(results_1.keys())


F_fs = []
F_ns = []
for key in keys:
    F_fs += [[results_1[key][0][2],results_2[key][0][2]]]
    F_ns += [[results_1[key][1][2],results_2[key][1][2]]]

F_fs = np.array(F_fs)
F_ns = np.array(F_ns)



# ###### PLOT ABSOLUTE VALUES
# ## Rank by baseline framewise F0
# print(F_fs.shape)
# indexes = np.argsort(F_fs[:,1])
# F_fs_sort = F_fs[indexes]
# keys_1 = [keys[i] for i in indexes]
#
# indexes = np.argsort(F_ns[:,1])
# F_ns_sort = F_ns[indexes]
# keys_2 = [keys[i] for i in indexes]
#
# fig, (ax1,ax2) = plt.subplots(2,1,figsize=[14,7])
#
# index = np.arange(len(keys))
# bar_width = 0.15
#
# opacity = 1
#
# rects1 = ax1.bar(index, F_fs_sort[:,0], bar_width,
#                 alpha=opacity, color='darkblue',
#                 label='Frame, MLM')
# rects2 = ax1.bar(index + bar_width, F_fs_sort[:,1], bar_width,
#                 alpha=opacity, color='lightblue',
#                 label='Framewise, BL')
# ax1.set_title('Scores by piece and model, '+folder_1+' vs '+folder_2)
# ax1.set_xticks(index + bar_width / 2)
# ax1.set_xticklabels(keys_1,fontsize=5,rotation=90)
# ax1.legend(prop={'size': 4})
#
#
# rects3 = ax2.bar(index, F_ns_sort[:,0], bar_width,
#                 alpha=opacity, color='darkred',
#                 label='Note, MLM')
# rects4 = ax2.bar(index + bar_width, F_ns_sort[:,1], bar_width,
#                 alpha=opacity, color='pink',
#                 label='Note, BL')
# ax2.set_title('F-measure by piece and model')
# ax2.set_xticks(index + bar_width / 2)
# ax2.set_xticklabels(keys_2,fontsize=5,rotation=90)
# ax2.legend(prop={'size': 4})
#
# fig.tight_layout()
# plt.show()


#### PLOT DIFFERENCES
# Diff_f = F_fs[:,0]-F_fs[:,1]
# Diff_n = F_ns[:,0]-F_ns[:,1]
#
# indexes = np.argsort(Diff_f)
# Diff_f = Diff_f[indexes]
# Diff_n = Diff_n[indexes]
# keys_1 = [keys[i] for i in indexes]
#
# fig, (ax1) = plt.subplots(1,1,figsize=[14,7])
#
# index = np.arange(len(keys))
# bar_width = 0.15
#
# opacity = 1
#
# rects1 = ax1.bar(index, Diff_f, bar_width,
#                 alpha=opacity, color='blue',
#                 label='Diff, frame')
# rects2 = ax1.bar(index + bar_width, Diff_n, bar_width,
#                 alpha=opacity, color='red',
#                 label='Diff, note')
# ax1.set_title('Difference in F-measure by piece and model, '+folder_1+' vs '+folder_2)
# ax1.set_xticks(index + bar_width / 2)
# ax1.set_xticklabels(keys_1,fontsize=5,rotation=90)
# ax1.legend(prop={'size': 4})
# fig.tight_layout()
# plt.show()


#### SCATTER : DIFF vs BASELINE F-MEASURE
Diff_f = F_fs[:,0]-F_fs[:,1]
Diff_n = F_ns[:,0]-F_ns[:,1]
BL_f = F_fs[:,1]
BL_n = F_ns[:,1]

fig, (ax1) = plt.subplots(1,1,figsize=[14,7])
ax1.scatter(BL_f,Diff_f,c='blue',label='Frame')
ax1.scatter(BL_n,Diff_n,c='red',label='Note')
ax1.set_title('Difference in F-measure against baseline F-measure, '+folder_1+' vs '+folder_2)
ax1.legend(prop={'size': 4})
fig.tight_layout()
plt.show()
