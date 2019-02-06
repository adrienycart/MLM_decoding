from weight_search import weight_search

import GPyOpt
import dill

domain = [{'name' : 'gt', 'type' : 'categorical', 'domain' : (True, False)},
          {'name' : 'min_diff', 'type' : 'continuous', 'domain' : (0, 1)},
          {'name' : 'history', 'type' : 'discrete', 'domain' : range(11)},
          {'name' : 'num_layers', 'type' : 'discrete', 'domain' : range(4)},
          {'name' : 'is_weight', 'type' : 'categorical', 'domain' : (True, False)},
          {'name' : 'features', 'type' : 'categorical', 'domain' : (True, False)}]

myBopt = GPyOpt.methods.BayesianOptimization(f=weight_search, domain=domain, maximize=True, verbosity=True, num_cores=4)

max_iter = 1000

myBopt.run_optimization(max_iter=max_iter, verbosity=True, report_file="optim/Report.txt", evaluations_file="optim/Evaluations.txt")

print(myBopt.x_opt)
print(myBopt.fx_opt)

with open("optim/optimized.dill", "wb") as file:
    dill.dump(myBopt, file)