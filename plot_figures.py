import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import scipy.stats


# folder_1 = 'results/quant/save-quant'
# folder_1 = "results/save-gtweight/save-quant-gtweight"
# folder_1 = "results/quant/sched0.7_w0.8_k50_b200_h12_new"
# folder_1 = "results/quant/w0.8_k40_b100_h20"
folder_1 = "results/results-20/save/wm-quant-sched"
folder_2 = 'results/results-20/save/wm-time-sched'
folder_BL = 'results/results-20/save/baseline-quant'
# folder_2 = 'results/baseline/quant'
# folder_2 = "results/quant/sched0.7_w0.8_k50_b200_h12"
# folder_2 = "results/save-hmm/save-quant-hmm"

print(folder_1)
print(folder_2)


try :
    results_1 = pickle.load(open(os.path.join(folder_1,'results.pkl'), "rb"))
except:
    results_1 = pickle.load(open(os.path.join(folder_1,'results.p'), "rb"))
try:
    results_2 = pickle.load(open(os.path.join(folder_2,'results_quant.pkl'), "rb"))
except:
    results_2 = pickle.load(open(os.path.join(folder_2,'results_quant.p'), "rb"))
# try :
#     results_2 = pickle.load(open(os.path.join(folder_2,'results.pkl'), "rb"))
# except:
#     results_2 = pickle.load(open(os.path.join(folder_2,'results.p'), "rb"))
try:
    results_BL = pickle.load(open(os.path.join(folder_BL,'results.pkl'), "rb"))
except:
    results_BL = pickle.load(open(os.path.join(folder_BL,'results.p'), "rb"))

keys = list(results_BL.keys())


print(len(keys))
F_fs = []
F_ns = []
for key in keys:
    try:
        F_fs += [[results_1[key][0][2],results_2[key][0][2],results_BL[key][0][2]]]
        F_ns += [[results_1[key][1][2],results_2[key][1][2],results_BL[key][1][2]]]
    except KeyError:
        print(key)
        keys.remove(key)

print(len(keys))
F_fs = np.array(F_fs)
F_ns = np.array(F_ns)

F_f_mean = np.mean(F_fs,axis=0)
F_n_mean = np.mean(F_ns,axis=0)

print(f"Framewise:  {F_f_mean[0]:.3f}, {F_f_mean[1]:.3f}, {F_f_mean[2]:.3f}")
print(f"Notewise:  {F_n_mean[0]:.3f}, {F_n_mean[1]:.3f}, {F_n_mean[2]:.3f}")
print("Frames: ",scipy.stats.ttest_rel(F_fs[:,1],F_fs[:,0]))
print("Notes: ",scipy.stats.ttest_rel(F_ns[:,1],F_ns[:,0]))


# ###### PLOT ABSOLUTE VALUES
## Rank by baseline framewise F0
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
# print(index.shape,F_fs_sort[:,0].shape)
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
# Diff_f = F_fs[:,0]-F_fs[:,1]
# Diff_n = F_ns[:,0]-F_ns[:,1]
# BL_f = F_fs[:,2]
# BL_n = F_ns[:,2]
#
# # from scipy.stats import linregress
# # slope_n, intercept_n, r_value_n, p_value_n, std_err_n = linregress(BL_f,F_fs[:,0])
# # slope_f, intercept_f, r_value_f, p_value_f, std_err_f = linregress(BL_n,F_ns[:,0])
#
# # print("Frame: Slope = ",slope_f ,"R value = ", r_value_f,"P value = ", p_value_f,"Standard error = ", std_err_f)
# # print("Note: Slope = ",slope_n ,"R value = ", r_value_n,"P value = ", p_value_n,"Standard error = ", std_err_n)
#
#
# fig, (ax1) = plt.subplots(1,1,figsize=[10,5])
# # ax1.scatter(BL_f,F_fs[:,0],c='blue',label='Frame')
# ax1.scatter(BL_n,Diff_n,c='black',label='Note')
# # ax1.set_title('Difference in F-measure against baseline F-measure, '+folder_1+' vs '+folder_2)
# # ax1.legend(prop={'size': 4})
# # ax1.set_xlim(0,1)
# # ax1.set_ylim(-0.15,0.15)
# ax1.plot([0,1],[0,0],linestyle=':',color='black')
# # ax1.set_xlabel('Baseline framewise F-measure',fontsize=10)
# # ax1.set_ylabel('On-notewise F-measure improvement from PM to PM+S',fontsize=10)
#
# ax1.tick_params(axis='both', which='major', labelsize=16)
# # ax1.tick_params(axis='both', which='minor', labelsize=2)
# # ax1.plot([0,1], np.multiply(slope_f,[0,1]) + np.full([2],intercept_f), color='blue',linestyle='--')
# # ax1.plot([0,1], np.multiply(slope_n,[0,1]) + np.full([2],intercept_n), color='red',linestyle='--')
# # ax1.text(0, 0.10, f"Frame: R value = {r_value_f:.3f}, P value = {p_value_f:.3f}, Standard error = {std_err_f:.3f}\n"+\
#                             # f"Note: R value = {r_value_n:.3f}, P value = {p_value_n:.3f}, Standard error = {std_err_n:.3f}")
# fig.tight_layout()
# plt.show()
