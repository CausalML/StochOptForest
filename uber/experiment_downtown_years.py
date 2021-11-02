from tree import *
from cvar_tree_utilities import *
import mkl
mkl.set_num_threads(1)

year_list = ["twoyear", "oneyear", "onehalfyear", "halfyear"]
X_list = {y: pd.read_csv("../X_" + str(y) + ".csv") for y in year_list}
Y_list = {y: pd.read_csv("../Y_" + str(y) + ".csv") for y in year_list}
A_mat = pd.read_csv("../A_downtwon_1221to1256.csv").to_numpy()
b_vec = pd.read_csv("../b_downtwon_1221to1256.csv").to_numpy()


seed = 0
np.random.seed(seed)


runs = 50
n_jobs = 50

alpha = 0.8
min_leaf_size = 10
max_depth = 100
n_proposals = 365
mtry = 65
honesty = False
balancedness_tol = 0.2
lb = 0; ub = 1
verbose = False
bootstrap = True;
n_trees = 100
subsample_ratio = 1

direct = ''
date = ''
output = direct + date + "experiment_downtown_years.txt"
with open(output, 'w') as f:
    print("start", file = f)

models_forest = {}
times_forest = {}
decisions_forest = {}
risks_forest = {}
feature_split_all = {}
feature_importance_all = {}

for year in year_list:
    with open(output, 'a') as f:
        print("year: ", year, file = f)

    X = X_list[year]
    Y = Y_list[year]

    enc = OneHotEncoder()
    enc.fit(X[["Period"]])
    tf = enc.transform(X[["Period"]]).toarray()
    X[["AM", "EarlyMorning", "Evening", "Midday", "PM"]] = tf
    X[["AM", "EarlyMorning", "Evening", "Midday", "PM"]] = X[["AM", "EarlyMorning", "Evening", "Midday", "PM"]].astype(int)

    sss = StratifiedShuffleSplit(n_splits = runs, test_size = 0.5, random_state = seed)
    sss.get_n_splits(range(X["Period"].shape[0]), X[["Period", "weekday"]])
    split_index = sss.split(range(X["Period"].shape[0]), X[["Period", "weekday"]])

    X.drop(["Period"], inplace = True, axis = 1)
    X_train_list = []; X_test_list = [];
    Y_train_list = []; Y_test_list = []
    for train_index, test_index in split_index:
        X_train_list.append(X.loc[train_index, ].to_numpy())
        X_test_list.append(X.loc[test_index, ].to_numpy())
        Y_train_list.append(Y.loc[train_index, ].to_numpy())
        Y_test_list.append(Y.loc[test_index, ].to_numpy())
    p = X_train_list[0].shape[1]

    time0 = time.time()
    results_fit = Parallel(n_jobs=n_jobs, verbose = 3)(delayed(experiment_downtown_years)(Y_train_list[run], X_train_list[run], 
            Y_test_list[run], X_test_list[run], 
            A_mat = A_mat, b_vec = b_vec, alpha = alpha, ub = ub, lb = lb, 
            subsample_ratio = subsample_ratio, bootstrap = bootstrap, n_trees = n_trees, honesty = honesty, mtry = mtry, 
            min_leaf_size = min_leaf_size, max_depth = max_depth, n_proposals = n_proposals, 
                balancedness_tol = balancedness_tol, verbose = verbose, seed = seed) for run in range(n_jobs))
    models_forest[year] = [res[0] for res in results_fit]
    times_temp = [res[1] for res in results_fit]
    times_forest[year] = pd.concat([pd.DataFrame(tt, index = [0]) for tt in times_temp])
    with open(output, 'a') as f:
        print("training time: ", time.time()-time0, file = f)
        print("time: ", times_forest[year].mean(), file = f)
        print("------------------------", file = f)

    time0 = time.time()
    evaluations = Parallel(n_jobs=n_jobs, verbose = 3)(delayed(evaluate_one_run)(models_forest[year][run], X_train_list[run], Y_train_list[run], 
                      X_train_list[run], Y_train_list[run],
                      X_test_list[run], Y_test_list[run],  
                     A_mat = A_mat, b_vec = b_vec, alpha = alpha) for run in range(n_jobs))
    decisions_forest[year] = [eval[0] for eval in evaluations]
    risks_temp = [eval[1] for eval in evaluations]
    risks_forest[year] = pd.concat([pd.DataFrame(rr, index = [0]) for rr in risks_temp])
    with open(output, 'a') as f:
        print("evaluation time:", time.time()-time0, file = f)
        print("risk: ", risks_forest[year].mean(), file = f)
        print("------------------------", file = f)

    pickle.dump(risks_forest[year], open(direct + date +  "downtown_risks_forest_years_"+year+".pkl", "wb"))

    feature_split_freq = evaluate_feature_split_freq(models_forest[year], p)
    feature_split_all[year]  = feature_split_freq

    feature_importance = evaluate_feature_importance(models_forest[year], p)
    feature_importance_all[year] = feature_importance

    pickle.dump(feature_split_all[year], open(direct + date +  "feature_split_years_"+year+".pkl", "wb"))
    pickle.dump(feature_importance_all[year], open(direct + date +  "feature_imp_years_"+year+".pkl", "wb"))

    


