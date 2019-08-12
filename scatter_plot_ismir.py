import numpy as np
import pickle
import scipy.stats
import matplotlib.pyplot as plt

with open("results/results-20/save/pm-quant/results.p", "rb") as file:
    pm = pickle.load(file)

with open("results/results-20/save/pm-quant-sched/results.p", "rb") as file:
    pm_s = pickle.load(file)

with open("results/results-20/save/baseline-quant/results.p", "rb") as file:
    baseline = pickle.load(file)

pm_scores = []
pm_s_scores = []
baseline_scores = []

for key in pm_s:
    pm_scores.append(pm[key][1][2])
    pm_s_scores.append(pm_s[key][1][2])
    baseline_scores.append(baseline[key][1][2])

pm_scores = np.array(pm_scores) * 100
pm_s_scores = np.array(pm_s_scores) * 100
baseline_scores = np.array(baseline_scores) * 100

from scipy.stats import linregress
slope_n, intercept_n, r_value_n, p_value_n, std_err_n = linregress(baseline_scores,pm_s_scores - pm_scores)

print("Note: Slope = ",slope_n ,"R value = ", r_value_n,"P value = ", p_value_n,"Standard error = ", std_err_n)
print('R2 = ', r_value_n**2)

plt.rcParams.update({'font.size': 13.5, 'font.family' : 'serif'})
plt.rcParams["figure.figsize"] = [6.5,3.8]
plt.scatter(baseline_scores, pm_s_scores - pm_scores, s=8, color="black")
plt.plot([0, 100], [0, 0],  color='grey', linewidth=0.5)
plt.plot([0,100], np.multiply(slope_n,[0,100]) + np.full([2],intercept_n), color='black',linestyle=':')

plt.xlabel("[13] On-notewise F-measure")
plt.ylabel("On-notewise F-measure increase")
plt.tight_layout()
# plt.show()
plt.savefig("scatter.png",bbox_inches='tight')
