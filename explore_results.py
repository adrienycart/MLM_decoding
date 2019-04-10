import os
import pickle


# Good comparisons:
# MAPS_MUS-chpn-p4_ENSTDkAm.mid: very good for WM, less good for PM
# MAPS_MUS-chpn_op35_1_ENSTDkAm.mid: our models get the rhythm right


path_baseline = 'results/results-20/save/baseline-quant/results.p'
path_hmm = 'results/results-20/save/hmm-quant/results.pkl'
path_pm = 'results/results-20/save/pm-quant/results.p'
path_wm = 'results/results-20/save/wm-quant/results.p'

all_results = []
for path in [path_baseline,path_hmm,path_pm,path_wm]:
    with open(path, "rb") as file:
        result = pickle.load(file)
        all_results += [[path,result]]


keys = all_results[0][1].keys()


###### PAIRWISE MODEL COMPARISON MAX and MIN
# for i,(path_1,result_1) in enumerate(all_results):
#     for (path_2,result_2) in all_results[i+1:]:
#         diff = []
#         for key in keys:
#             diff += [[key,result_1[key][1][-1]-result_2[key][1][-1]]]
#
#         print(path_1)
#         print(path_2)
#         print(max(diff,key=lambda x:x[1]))
#         print(min(diff,key=lambda x:x[1]))


####### MAX ABS DIFF ACROSS MODELS
# all_diffs = []
# for key in keys:
#     diff = 0
#     for i,(path_1,result_1) in enumerate(all_results):
#         for (path_2,result_2) in all_results[i+1:]:
#             diff+= abs(result_1[key][1][-1]-result_2[key][1][-1])
#     all_diffs += [[key,diff]]
#
# sort = sorted(all_diffs,key=lambda x:x[1])
# for filename,value in sort[-10:]:
#     print(filename,value)

####### MAX DIFF IN FAVOUR OF OUR MODELS
all_diffs = []
for key in keys:
    diff = 0
    diff+= all_results[2][1][key][1][-1]-all_results[0][1][key][1][-1]
    diff+= all_results[3][1][key][1][-1]-all_results[0][1][key][1][-1]
    all_diffs += [[key,diff]]
sort = sorted(all_diffs,key=lambda x:x[1])
for filename,value in sort[-10:]:
    print(filename,value)
