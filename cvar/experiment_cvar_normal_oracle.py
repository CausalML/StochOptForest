from tree import *
from cvar_tree_utilities import *
import mkl
mkl.set_num_threads(1)

p = 10
n_jobs = 50
runs = 50
n_trees = 50;
Nx_test = 200
Ny_train = 1000
Ny_test = 2000
R = 0.1
alpha = 0.1
obj_coef_list = [0]
N_list = [100, 200, 400]
lb = 0; ub = 1;  sum_bound = 1; if_stoch_constr = False


generate_Y = generate_Y_normal

honesty = False;
verbose = False; oracle = True;
bootstrap = True; 

cond_mean = [lambda x: np.exp(x[:, 0]), lambda x: x[:, 0], lambda x: np.abs(x[:, 0])]
cond_std = [lambda x: 5 - 4*((-3<=x[:, 1]) & (x[:, 1]<=-1)), lambda x: 5 - 4*((-1<=x[:, 1])&(x[:, 1]<=1)), lambda x: 5 - 4*((1<=x[:, 1])&(x[:, 1]<=3))]

risk_all = {}
feature_split_all = {}
results_eval_all = {}

direct = ''
date = ''
output = "cvar_normal_oracle.txt"

with open(output, 'w') as f:
    print("start", file = f)

for N in N_list:
    risk_all[str(N)] = {}
    feature_split_all[str(N)] = {}
    results_eval_all[str(N)] = {}

    for obj_coef in obj_coef_list:

        n_proposals = N; 
        mtry = p;
        subsample_ratio = 1;
        max_depth=100; 
        min_leaf_size=10; 
        balancedness_tol = 0.2; 

        X_list = [np.random.normal(size = (N, p)) for run in range(runs)]
        Y_list = [generate_Y(X_list[run], cond_mean, cond_std) for run in range(runs)]

        with open(output, 'a') as f:
            print("N: ", N, file = f)
            print("obj_coef: ", obj_coef, file = f)

        time1 = time.time()
        results_fit = Parallel(n_jobs=n_jobs, verbose = 3)(delayed(compare_forest)(X_list[run], Y_list[run], X_list[run], Y_list[run], 
            alpha = alpha, R = R, obj_coef = obj_coef, lb = lb, ub = ub,  sum_bound = sum_bound, if_stoch_constr = if_stoch_constr,
            n_trees = n_trees, honesty= honesty, mtry = mtry, subsample_ratio = subsample_ratio, oracle = oracle, min_leaf_size = min_leaf_size, 
            verbose = verbose, max_depth = max_depth, n_proposals = n_proposals, balancedness_tol = balancedness_tol, bootstrap = bootstrap) for run in range(runs))
        time2 = time.time()
        with open(output, 'a') as f:
            print("time: ", time2 - time1, file = f)
            print("------------------------", file = f)

        time1 = time.time()
        results_eval_all[str(N)][str(obj_coef)] = Parallel(n_jobs=n_jobs, verbose = 3)(delayed(evaluate_one_run_determ_constr)(results_fit[run], X_list[run], Y_list[run], X_list[run], Y_list[run], 
            Nx_test,Ny_train, Ny_test, cond_mean, cond_std, 
            alpha = alpha, R = R, obj_coef = obj_coef, lb = lb, ub = ub,  sum_bound = sum_bound, if_stoch_constr = if_stoch_constr,
            verbose = False, generate_Y = generate_Y) for run in range(runs))
        time2 = time.time()
        with open(output, 'a') as f:
            print("time: ", time2 - time1, file = f)
            print("------------------------", file = f)

        risks = extract_risk(results_eval_all[str(N)][str(obj_coef)])
        with open(output, 'a') as f:
            print("risk:", N, file=f)
            for k,v in sorted(risks.items(), key = lambda x: x[1].mean()):
                print(k,"avg risk:",np.mean(v),"+-", 2*np.std(v)/np.sqrt(len(v)), file=f)
            print("------------------------", file = f)
        risk_all[str(N)][str(obj_coef)] = risks

        feature_split_freq = evaluate_feature_split_freq(results_fit, p)
        with open(output, 'a') as f:
            for k,v in sorted(feature_split_freq.items(), key = lambda x: x[1].mean(0)[0], reverse = True):
                print(k,"frac feat. slt.:",v.mean(0), file = f)
            print("---", file = f)
            print("----------------------", file = f)
            print("----------------------", file = f)
        feature_split_all[str(N)][str(obj_coef)] = feature_split_freq

        pickle.dump(results_eval_all, open(direct + date +  "results_eval_cvar_normal_oracle.pkl", "wb"))
        pickle.dump(risk_all, open(direct + date +  "risk_cvar_normal_oracle.pkl", "wb"))
        pickle.dump(feature_split_all, open(direct + date +  "feature_split_cvar_normal_oracle.pkl", "wb"))
        


