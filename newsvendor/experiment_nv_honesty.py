from tree import *
from nv_tree_utilities import *

import mkl
mkl.set_num_threads(1)

# forest 

p_list = [10]
runs = 50
n_jobs = 50
n_trees = 500;
N_list = [100, 200, 400, 800]
Nx_test = 200
Ny_test = 2000
Ny_train = 1000



b_list = np.array([100., 1.])
h_list = np.array([5., 0.05])
C = 1000
L = len(h_list)

honesty = False; 
verbose = False; oracle = True;
bootstrap = False; 

cond_mean = [lambda x: 3, lambda x: 3]
cond_std = [lambda x: np.exp(x[:, 0]), lambda x: np.exp(x[:, 1])]

risk_all = {}
feature_split_all = {}
results_eval_all = {}

direct = ''
date = ''
output = "nv_honesty.txt"

with open(output, 'w') as f:
    print("start", file = f)

for N in N_list:
    risk_all[str(N)] = {}
    feature_split_all[str(N)] = {}
    results_eval_all[str(N)] = {}

    for p in p_list:
        with open(output, 'a') as f:
            print("N: ", N, file = f)
            print("p: ", p, file = f)

        n_proposals = N; 
        mtry = p;
        subsample_ratio = 0.63;
        max_depth=100; 
        min_leaf_size=10; 
        balancedness_tol = 0.2; 

        X_list = [np.random.normal(size = (N, p)) for run in range(runs)]
        Y_list = [generate_Y(X_list[run], cond_mean, cond_std) for run in range(runs)]
    
        time1 = time.time()
        results_fit = Parallel(n_jobs=n_jobs, verbose = 3)(delayed(compare_forest_one_run_honesty)(X_list[run], Y_list[run], X_list[run], Y_list[run], 
            h_list = h_list, b_list = b_list, C = C, 
            n_trees = n_trees, honesty= honesty, mtry = mtry, subsample_ratio = subsample_ratio, oracle = oracle, min_leaf_size = min_leaf_size, verbose = verbose, max_depth = max_depth, n_proposals = n_proposals, balancedness_tol = balancedness_tol, bootstrap = bootstrap) for run in range(runs))
        time2 = time.time()
        with open(output, 'a') as f:
            print("time: ", time2 - time1, file = f)
            print("------------------------", file = f)

        time1 = time.time()
        results_eval = Parallel(n_jobs=n_jobs, verbose = 3)(delayed(evaluate_one_run)(results_fit[run], X_list[run], Y_list[run], X_list[run], Y_list[run], 
            Nx_test, Ny_train, Ny_test, cond_mean, cond_std,  
            h_list =h_list, b_list = b_list, C = C, verbose = verbose) for run in range(runs))
        time2 = time.time()
        results_eval_all[str(N)][str(p)] = results_eval
        with open(output, 'a') as f:
            print("time: ", time2 - time1, file = f)
            print("------------------------", file = f)

        risks = extract_risk(results_eval)
        with open(output, 'a') as f:
            print("risk with C", C, file=f)
            for k,v in sorted(risks.items(), key = lambda x: x[1].mean()):
                print(k,"avg risk:",np.mean(v),"+-", 2*np.std(v)/np.sqrt(len(v)), file=f)
            print("------------------------", file = f)
        risk_all[str(N)][str(p)] = risks

        feature_split_freq = evaluate_feature_split_freq(results_fit, p)
        with open(output, 'a') as f:
            for k,v in sorted(feature_split_freq.items(), key = lambda x: x[1].mean(0)[0], reverse = True):
                print(k,"frac feat. slt.:",v.mean(0), file = f)
            print("---", file = f)
            print("----------------------", file = f)
            print("----------------------", file = f)
        feature_split_all[str(N)][str(p)] = feature_split_freq

        pickle.dump(risk_all, open(direct + date +  "risk_nv_honesty.pkl", "wb"))
        pickle.dump(feature_split_all, open(direct + date +  "feature_split_nv_honesty.pkl", "wb"))
        pickle.dump(results_eval_all, open(direct + date +  "results_eval_nv_honesty.pkl", "wb"))