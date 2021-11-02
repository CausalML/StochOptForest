import numpy as np
import warnings
from scipy.stats import gaussian_kde
import time
# from sklearn.utils import check_random_state
from joblib import Parallel, delayed
from functools import partial
from functools import reduce
from statsmodels.nonparametric.kernel_regression import KernelReg
import pickle 
import scipy.stats 
from scipy.stats import truncnorm
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
from tree import *


import mkl
mkl.set_num_threads(1)

def generate_Y_normal(X, cond_mean, cond_std, Ny = None):
    L = len(cond_std)
    
    if Ny is None:
        Ny = X.shape[0]

    Y = np.zeros((Ny, L))
    for l in range(L):
        Y[:, l] = cond_mean[l](X) + np.random.normal(0, cond_std[l](X), Ny)
        
    return Y

def generate_Y_lognormal(X, cond_mean, cond_std, Ny = None):
    L = len(cond_std)
    
    if Ny is None:
        Ny = X.shape[0]

    Y = np.zeros((Ny, L))
    for l in range(L):
        Y[:, l] = cond_mean[l](X) - np.random.lognormal(0, cond_std[l](X), Ny)  
        
    return Y

def solve_mean_variance(Y, alpha = None, R = None, obj_coef = None, lb = 0, ub = 1, sum_bound = 1, verbose = False, weights = None, if_weight = False, if_stoch_constr=False):
    
    L = Y.shape[1]
    n = Y.shape[0]

    if not if_weight:
        weights = np.ones(n)/n

    mu = np.sum(Y * weights[:, np.newaxis], 0)
    sigma = np.matmul(np.matmul(np.transpose(Y - mu), np.diag(weights)), Y - mu)

    m = gp.Model()
        
    z = pd.Series(m.addVars(L, lb = lb, ub = ub, name = 'w'), index = range(L))
    m.update()
    avg_return = z.dot(mu)
    risk = np.dot(np.dot(sigma, z), z) - obj_coef * avg_return

    m.setObjective(risk, GRB.MINIMIZE) 
    constr_budget = m.addConstr(z.sum() == 1.0, 'budget')
    if if_stoch_constr:
        constr_return = m.addConstr(avg_return >= R, 'return')

    m.update()

    if not verbose:
        m.setParam('OutputFlag', 0)
        
    if verbose:
        print("sigma", sigma)
        print("mu", mu)

    try:
        m.optimize()
        decision = np.zeros(len(z)+1)
        decision[:-1] = np.array([zz.X for zz in z])
        decision[-1] = decision[:-1].dot(mu)
        lam_de = constr_budget.Pi
        lam_st = 0
        if if_stoch_constr:
            lam_st = constr_return.Pi
        obj = m.objVal
    except:
        if verbose:
            print("optimization error!")
        decision = None
        lam_de = None
        lam_st = None
        obj = np.inf

    return (decision, lam_de, lam_st, obj)

###############################
# criterion
###############################
def compute_crit_oracle(node_Y, node_sol, node_h_left, node_h_right, f_grad, g_grad, node_hessian, nu0, lambda0, valid_side, solver = None):
    
    def compute_oracle_risk(node_Y, split_obs):
        temp_risk = np.zeros(split_obs.shape[1])
        for l in range(split_obs.shape[1]):
            Ytemp = node_Y[split_obs[:, l], :]
            (_, _, _, temp_risk[l]) = solver(Ytemp)
        return temp_risk
    
    crit_left = compute_oracle_risk(node_Y, valid_side)
    crit_right= compute_oracle_risk(node_Y, ~valid_side)
    crit = valid_side.sum(0)/node_Y.shape[0] * crit_left + (1 - valid_side).sum(0)/node_Y.shape[0] * crit_right

    if (crit != np.inf).sum() == 0:
        return (None, None, None)
    else:
        return (crit, crit_left, crit_right)

def compute_crit_random(node_Y, node_sol, node_h_left, node_h_right, f_grad, g_grad, node_hessian, nu0, lambda0, valid_side):
    return (np.random.rand(valid_side.shape[1]), None, None)

def compute_crit_rf(node_Y, node_sol, node_h_left, node_h_right, f_grad, g_grad, node_hessian, nu0, lambda0, valid_side):

    def compute_crit_rf(node_Y, split_obs):
        temp_n = np.sum(split_obs, 0)
        mean_temp = np.matmul(np.transpose(node_Y), split_obs)/temp_n
        SS = np.zeros((node_Y.shape[1], split_obs.shape[1]))
        for l in range(node_Y.shape[1]):
            SS[l, :] = np.sum(((node_Y[:, l][:, np.newaxis] - mean_temp[l, :][np.newaxis, :]) ** 2) * split_obs, 0)
        return SS

    crit_left = compute_crit_rf(node_Y, valid_side).sum(0)
    crit_right = compute_crit_rf(node_Y, ~valid_side).sum(0)
    crit = crit_left+ crit_right

    return (crit, crit_left, crit_right)

def compute_crit_approx_risk(node_Y, node_sol, node_h_left, node_h_right, f_grad, g_grad, node_hessian, nu0, lambda0, valid_side):

    def compute_risk_approx_risk(node_Y, node_h, f0_grad, fj_grad, g0_grad, gj_grad, node_hessian, nu0, lambda0):
        term1 = np.diagonal(np.matmul(np.matmul(np.transpose(node_h), node_hessian), node_h))
        term2 = 2 * np.diag(np.matmul(np.transpose(node_h), fj_grad - f0_grad + (gj_grad - g0_grad) * lambda0))

        return (term1 + term2) 

    crit_left = compute_risk_approx_risk(node_Y, node_h_left, f_grad['0'], f_grad['j_left'], g_grad['0'], g_grad['j_left'], node_hessian, nu0, lambda0)
    crit_right = compute_risk_approx_risk(node_Y, node_h_right, f_grad['0'], f_grad['j_right'], g_grad['0'], g_grad['j_right'], node_hessian, nu0, lambda0)
    
    crit = valid_side.sum(0)/node_Y.shape[0] * crit_left + (1 - valid_side).sum(0)/node_Y.shape[0] * crit_right
    
    return (crit, crit_left, crit_right)

def compute_crit_approx_sol(node_Y, node_sol, node_h_left, node_h_right, f_grad, g_grad, node_hessian, nu0, lambda0, valid_side, obj_coef = None, alpha = None):
    
    def compute_risk_approx_sol(node_Y, node_sol, node_h, split_obs):
        sol_temp = node_sol[:, np.newaxis] + node_h

        temp1 = np.matmul(node_Y, sol_temp[0:(sol_temp.shape[0]-1), :])  # Y^Tz_{1:d}
        temp2 = sol_temp[-1, :][np.newaxis, :]  # z_{d+1}
        obj_temp = np.diag(np.matmul(np.transpose(split_obs), np.power(temp1 - temp2, 2)))/split_obs.sum(0)

        return obj_temp

    crit_left = compute_risk_approx_sol(node_Y, node_sol, node_h_left, valid_side)
    crit_right = compute_risk_approx_sol(node_Y, node_sol, node_h_right, ~valid_side)

    crit = valid_side.sum(0)/node_Y.shape[0] * crit_left + (1 - valid_side).sum(0)/node_Y.shape[0] * crit_right
    
    return (crit, crit_left, crit_right)

def search_active_constraint(node_Y, node_sol, R = None, lb = 0, ub = 1, sum_bound = 1, verbose = False, if_stoch_constr = False):
    # assert that node_sol >= 0, node_lambda >= 0
    L = node_Y.shape[1]
    active_const_de = np.array([False]*(1 + 2 * L))
    active_const_st = np.array([False]) 

    if np.abs(node_sol[:-1].sum() - sum_bound) < 1e-4:
        active_const_de[0] = True
    active_const_de[1:(L+1)] = np.abs(node_sol[:-1] - lb) < 1e-4
    active_const_de[(L+1):] = np.abs(node_sol[:-1] - ub) < 1e-4

    if if_stoch_constr:
        if np.abs(np.mean(node_Y, 0).dot(node_sol[:-1]) - R) < 1e-4:
            active_const_st[0] = True
        
    return (active_const_de, active_const_st)

def compute_gradient(node_Y, node_sol, alpha = None, R = None, obj_coef = None):

    # np.matmul(node_Y, node_sol[0:node_Y.shape[1]]) is a column veccor of [Y_1^t z_{1:d}, ..., Y_n^t z_{1:d}]
    # np.transpose(node_Y) is a matrix whose ith column is Y_i, i.e., np.transpose(node_Y) = [Y_1, ..., Y_n]
    # the following gives [Y_1Y_1^t z_{1:d}, ..., Y_nY_n^t z_{1:d}]
    term_upp1 = np.matmul(np.transpose(node_Y), np.diag(np.matmul(node_Y, node_sol[0:node_Y.shape[1]])))
    # z_{d+1} * [Y_1, ..., Y_n]
    term_upp2 = np.transpose(node_Y) * node_sol[-1]

    term_low1 = node_sol[-1]
    term_low2 = np.matmul(node_sol[np.newaxis, 0:node_Y.shape[1]], np.transpose(node_Y))

    # (L + 1) * n 
    obj_grad = 2 * np.concatenate((term_upp1 - term_upp2 - 1/2 * obj_coef * np.transpose(node_Y), term_low1 - term_low2), axis = 0)


    box_constr_gradient = np.concatenate((np.diag(np.ones(node_Y.shape[1])), np.zeros((1, node_Y.shape[1]))), axis= 0)
    constr_grad_de = np.array([1] * node_Y.shape[1] + [0])[:, np.newaxis]
    constr_grad_de = np.concatenate((constr_grad_de, -box_constr_gradient, box_constr_gradient), axis = 1)
    # (L + 1) * n 
    constr_grad_st = -np.concatenate((np.transpose(node_Y), np.zeros((1, node_Y.shape[0]))), axis = 0)

    return (obj_grad, constr_grad_de, constr_grad_st)

def compute_hessian(node_Y, node_sol,  alpha = None):

    term_upp = np.concatenate((np.matmul(np.transpose(node_Y), node_Y)/node_Y.shape[0], -node_Y.mean(0)[:, np.newaxis]), axis = 1)
    term_low = np.concatenate((-node_Y.mean(0)[np.newaxis, :], np.array([1])[np.newaxis, :]), axis = 1)

    return 2 * np.concatenate((term_upp, term_low), axis = 0)

def compute_update_step(node_Y, node_sol, nu0, lambda0, 
                                      node_hessian, 
                                      node_obj_gradient, node_constr_gradient_de, node_constr_gradient_st,
                                      active_const_de, active_const_st,
                                      valid_side, 
                                      R = None,
                                      constraint = True):
    
    def compute_child_gradient(node_obj_gradient, node_constr_gradient_st, split_obs):

        fj_grad = np.matmul(node_obj_gradient, split_obs)/split_obs.sum(0)
        gj_grad = np.matmul(node_constr_gradient_st, split_obs)/split_obs.sum(0)

        return (fj_grad, gj_grad)

    def compute_full_lhs(node_hessian, node_constr_gradient_st, node_constr_gradient_de):
        L22 = node_hessian
        G_grad = node_constr_gradient_st.mean(1)[:, np.newaxis]
        H_grad = node_constr_gradient_de

        lhs_upper = np.concatenate((2*L22, G_grad, H_grad), axis = 1)
        lhs_middle = np.concatenate((np.transpose(G_grad), np.zeros((G_grad.shape[1], G_grad.shape[1])), np.zeros((G_grad.shape[1], H_grad.shape[1]))), axis = 1)
        lhs_lower = np.concatenate((np.transpose(H_grad), np.zeros((H_grad.shape[1], G_grad.shape[1])), np.zeros((H_grad.shape[1], H_grad.shape[1]))), axis = 1)
        lhs_full = np.concatenate((lhs_upper, lhs_middle, lhs_lower), axis = 0)

        return lhs_full

    def compute_full_rhs(node_Y, node_sol, f0_grad, g0_grad, 
                         node_obj_gradient, node_constr_gradient_st, 
                         nu0, lambda0, split_obs, R):
        (fj_grad, gj_grad) = compute_child_gradient(node_obj_gradient, node_constr_gradient_st, split_obs)

        # G is a 1 * n array
        G = R - np.matmul(node_Y, node_sol[:-1])[np.newaxis, :]
        G0 = G.mean(1)[:, np.newaxis]
        Gj = np.matmul(G, split_obs)/split_obs.sum(0)

        rhs_upper = -2 * (fj_grad - f0_grad) - 2 * (gj_grad - g0_grad) * lambda0
        rhs_middle = - (Gj - G0)
        rhs_lower = np.zeros((node_constr_gradient_de.shape[1], split_obs.shape[1]))
        rhs = np.concatenate((rhs_upper, rhs_middle, rhs_lower), axis = 0)

        return (rhs, fj_grad, gj_grad)

    def compute_update(lhs, rhs):
        try:
            h = np.linalg.solve(lhs, rhs)[0:(len(node_sol)), :]
        except:
            lhs = lhs + np.diag([1e-3]*lhs.shape[1])
            h = np.linalg.solve(lhs, rhs)[0:(len(node_sol)), :]

        return h 

    def compute_update_step_constr(lhs, node_Y, node_sol, f0_grad, g0_grad, 
                         node_obj_gradient, node_constr_gradient_st, 
                         nu0, lambda0, split_obs, R, index):
        (rhs, fj_grad, gj_grad) = compute_full_rhs(node_Y, node_sol, f0_grad, g0_grad, 
                         node_obj_gradient, node_constr_gradient_st, 
                         nu0, lambda0, split_obs, R)
        rhs = rhs[index, :]
        node_h = compute_update(lhs, rhs)

        return (node_h, fj_grad, gj_grad)

    def compute_update_step_unconstr(node_obj_gradient, node_constr_gradient_st, node_hessian, split_obs):
        lhs = node_hessian
        (fj_grad, gj_grad) = compute_child_gradient(node_obj_gradient, node_constr_gradient_st, split_obs)
        rhs = - fj_grad 

        h = compute_update(lhs, rhs)

        return (h, fj_grad, gj_grad)
        
    if constraint:
        index = [True] * node_hessian.shape[0]  + list(active_const_st) + list(active_const_de)
        
        lhs_full = compute_full_lhs(node_hessian, node_constr_gradient_st, node_constr_gradient_de)
        lhs = lhs_full[:, index]
        lhs = lhs[index, :]
        
        f0_grad = node_obj_gradient.mean(1)[:, np.newaxis]
        g0_grad = node_constr_gradient_st.mean(1)[:, np.newaxis]
        
        (node_h_left, fj_grad_left, gj_grad_left) = compute_update_step_constr(lhs, node_Y, node_sol, f0_grad, g0_grad, 
                         node_obj_gradient, node_constr_gradient_st, 
                         nu0, lambda0, valid_side, R, index)
        (node_h_right, fj_grad_right, gj_grad_right) = compute_update_step_constr(lhs, node_Y, node_sol, f0_grad, g0_grad, 
                         node_obj_gradient, node_constr_gradient_st, 
                         nu0, lambda0, ~valid_side, R, index)
        f_grad = {'0': f0_grad, 'j_left': fj_grad_left, 'j_right': fj_grad_right}
        g_grad = {'0': g0_grad, 'j_left': gj_grad_left, 'j_right': gj_grad_right}

    else:
        f0_grad = node_obj_gradient.mean(1)[:, np.newaxis]
        g0_grad = node_constr_gradient_st.mean(1)[:, np.newaxis]

        (node_h_left, fj_grad_left, gj_grad_left) = compute_update_step_unconstr(node_obj_gradient, node_constr_gradient_st, node_hessian, valid_side)
        (node_h_right, fj_grad_right, gj_grad_right) = compute_update_step_unconstr(node_obj_gradient, node_constr_gradient_st, node_hessian, ~valid_side)
        f_grad = {'0': f0_grad, 'j_left': fj_grad_left, 'j_right': fj_grad_right}
        g_grad = {'0': g0_grad, 'j_left': gj_grad_left, 'j_right': gj_grad_right}

    return (node_h_left, node_h_right, f_grad, g_grad)

def compare_forest(X, Y, X_est, Y_est, 
    alpha = None, R = None, obj_coef = 0, lb = -1, ub = 1,  sum_bound = 1, if_stoch_constr = False,
    n_trees = 500, honesty = False, mtry = 5, verbose = False, subsample_ratio = 0.5, oracle = False,
    min_leaf_size=10, max_depth=100, n_proposals = 200, balancedness_tol = 0.3, bootstrap = False):
    models = {}

    opt_solver = partial(solve_mean_variance, alpha = alpha, R = R, obj_coef = obj_coef, lb = lb, ub = ub, sum_bound = sum_bound, if_stoch_constr = if_stoch_constr)
    hessian_computer = partial(compute_hessian, alpha = alpha)
    active_constraint = partial(search_active_constraint,  R = R, lb = lb, ub = ub, sum_bound = sum_bound, if_stoch_constr = if_stoch_constr)
    gradient_computer = partial(compute_gradient,  alpha = alpha, R = R, obj_coef = obj_coef)
    update_step = partial(compute_update_step, R = R)


    forest_temp = forest(opt_solver = opt_solver, hessian_computer = hessian_computer,
                         gradient_computer = gradient_computer, 
                          search_active_constraint = active_constraint,
                         compute_update_step = update_step,
                         crit_computer = partial(compute_crit_approx_sol, obj_coef = obj_coef, alpha = alpha),
                         subsample_ratio=subsample_ratio, bootstrap = bootstrap, n_trees = n_trees, 
                         honesty = honesty, mtry = mtry,
                         min_leaf_size = min_leaf_size, max_depth = max_depth, 
                         n_proposals = n_proposals, balancedness_tol = balancedness_tol,
                         verbose = verbose)
    forest_temp.fit(Y, X, Y_est, X_est)
    models['rf_approx_sol'] = forest_temp

    forest_temp = forest(opt_solver = opt_solver, hessian_computer = hessian_computer,
                         gradient_computer = gradient_computer, 
                          search_active_constraint = active_constraint,
                         compute_update_step = partial(update_step, constraint = False),
                         crit_computer = partial(compute_crit_approx_sol, obj_coef = obj_coef, alpha = alpha),
                        subsample_ratio=subsample_ratio, bootstrap = bootstrap, n_trees = n_trees, 
                         honesty = honesty, mtry = mtry,
                         min_leaf_size = min_leaf_size, max_depth = max_depth, 
                         n_proposals = n_proposals, balancedness_tol = balancedness_tol,
                         verbose = verbose)
    forest_temp.fit(Y, X, Y_est, X_est)
    models['rf_approx_sol_unconstr'] = forest_temp

    forest_temp = forest(opt_solver = opt_solver, hessian_computer = hessian_computer,
                         gradient_computer = gradient_computer, 
                          search_active_constraint = active_constraint,
                         compute_update_step = update_step,
                         crit_computer = compute_crit_approx_risk, 
                        subsample_ratio=subsample_ratio, bootstrap = bootstrap, n_trees = n_trees, 
                         honesty = honesty, mtry = mtry,
                         min_leaf_size = min_leaf_size, max_depth = max_depth, 
                         n_proposals = n_proposals, balancedness_tol = balancedness_tol,
                         verbose = verbose)
    forest_temp.fit(Y, X, Y_est, X_est)
    models['rf_approx_risk'] = forest_temp

    forest_temp = forest(opt_solver = opt_solver, hessian_computer = hessian_computer,
                         gradient_computer = gradient_computer, 
                          search_active_constraint = active_constraint,
                         compute_update_step = partial(update_step, constraint = False),
                         crit_computer = compute_crit_approx_risk, 
                        subsample_ratio=subsample_ratio, bootstrap = bootstrap, n_trees = n_trees, 
                         honesty = honesty, mtry = mtry,
                         min_leaf_size = min_leaf_size, max_depth = max_depth, 
                         n_proposals = n_proposals, balancedness_tol = balancedness_tol,
                         verbose = verbose)
    forest_temp.fit(Y, X, Y_est, X_est)
    models['rf_approx_risk_unconstr'] = forest_temp


    forest_temp = forest(opt_solver = opt_solver, hessian_computer = hessian_computer,
                         gradient_computer = gradient_computer, 
                          search_active_constraint = active_constraint,
                         compute_update_step = update_step,
                         crit_computer = compute_crit_random, 
                        subsample_ratio=subsample_ratio, bootstrap = bootstrap, n_trees = n_trees, 
                         honesty = honesty, mtry = mtry,
                         min_leaf_size = min_leaf_size, max_depth = max_depth, 
                         n_proposals = n_proposals, balancedness_tol = balancedness_tol,
                         verbose = verbose)
    forest_temp.fit(Y, X, Y_est, X_est)
    models['rf_random'] = forest_temp

    forest_temp = forest(opt_solver = opt_solver, hessian_computer = hessian_computer,
                         gradient_computer = gradient_computer, 
                          search_active_constraint = active_constraint,
                         compute_update_step = update_step,
                         crit_computer = compute_crit_rf, 
                        subsample_ratio=subsample_ratio, bootstrap = bootstrap, n_trees = n_trees, 
                         honesty = honesty, mtry = mtry,
                         min_leaf_size = min_leaf_size, max_depth = max_depth, 
                         n_proposals = n_proposals, balancedness_tol = balancedness_tol,
                         verbose = verbose)
    forest_temp.fit(Y, X, Y_est, X_est)
    models['rf_rf'] = forest_temp

    if oracle:
        forest_temp = forest(opt_solver = opt_solver, hessian_computer = hessian_computer,
                         gradient_computer = gradient_computer, 
                          search_active_constraint = active_constraint,
                         compute_update_step = update_step,
                         crit_computer = partial(compute_crit_oracle, solver = opt_solver), 
                        subsample_ratio=subsample_ratio, bootstrap = bootstrap, n_trees = n_trees, 
                         honesty = honesty, mtry = mtry,
                         min_leaf_size = min_leaf_size, max_depth = max_depth, 
                         n_proposals = n_proposals, balancedness_tol = balancedness_tol,
                         verbose = verbose)
        forest_temp.fit(Y, X, Y_est, X_est)
        models['rf_oracle'] = forest_temp

    return models

##########################
#  Evaluation
##########################
def evaluate_one_run_stoch_constr(models, X, Y, X_est, Y_est, Nx_test, Ny_train, Ny_test, cond_mean, cond_std, 
    alpha = None, R = None, obj_coef = None, lb = -1, ub = 1,  sum_bound = 1, if_stoch_constr = False, verbose = False,
    generate_Y = None):

    solver = partial(solve_mean_variance, alpha = alpha, obj_coef = 0, lb = lb, ub = ub, sum_bound = sum_bound, if_stoch_constr = if_stoch_constr)
    evaluate_risk = partial(evaluate_risk_singleX, obj_coef = 0)
    evluate_violation = partial(evaluate_constr_violation, R = R)

    p = X_est.shape[1]
    L = Y_est.shape[1]

    X_test = np.random.normal(size = (Nx_test, p))

    decisions = {}
    risk = {}
    violation = {}
    # risk_comb = {}

    for key in models.keys():
        decisions[key] = np.zeros((Nx_test, L))
        risk[key] = np.zeros(Nx_test)
        violation[key] = np.zeros(Nx_test)

        decisions[str(key) + "_oracle"] = np.zeros((Nx_test, L))
        risk[str(key) + "_oracle"] = np.zeros(Nx_test)
        violation[str(key) + "_oracle"] = np.zeros(Nx_test)

    decisions["oracle"] = np.zeros((Nx_test, L))
    violation["oracle"] = np.zeros(Nx_test)
    risk["oracle"] = np.zeros(Nx_test)

    for i in range(Nx_test):
        if i % 100 == 0:
                print("the", i, "th evaluation sample.")

        Y_train = generate_Y(X_test[i, :][np.newaxis, :], cond_mean, cond_std, Ny = Ny_train)
        Y_test = generate_Y(X_test[i, :][np.newaxis, :], cond_mean, cond_std, Ny = Ny_test)

        for key in models.keys():
            weights = models[key].get_weights(X_test[i, :])
            oracle_key = str(key) + "_oracle"
            try:
                (decision_temp, _, _, _) = solver(Y_est, R = R, weights = weights, if_weight = True)
                decisions[key][i, :] = decision_temp[:-1]
                risk[key][i] = evaluate_risk(decisions[key][i, :], Y_test)
                violation[key][i] = evluate_violation(decisions[key][i, :], Y_test)

                R0 = R - violation[key][i]
                (decision_temp, _, _, _) = solver(Y_train, R = R0, weights = None, if_weight = False)
                decisions[oracle_key][i, :] = decision_temp[:-1]
                risk[oracle_key][i] = evaluate_risk(decisions[oracle_key][i, :], Y_test)
                violation[oracle_key][i] =  evluate_violation(decisions[oracle_key][i, :], Y_test)
            except:
                decisions[key][i, :] = np.array([np.nan for i in range(len(decisions[key][i, :]))])
                risk[key][i] = np.nan
                violation[key][i] = np.nan
                decisions[oracle_key][i, :] = np.array([np.nan for i in range(len(decisions[oracle_key][i, :]))])
                risk[oracle_key][i] = np.nan
                violation[oracle_key][i] = np.nan

        key = "oracle"
        try:
            (decision_temp, _, _, _) = solver(Y_train, R = R, weights = None, if_weight = False)
            decisions[key][i, :] = decision_temp[:-1]
            risk[key][i] = evaluate_risk(decisions[key][i, :], Y_test)
            violation[key][i] = evluate_violation(decisions[key][i, :], Y_test)
        except:
            decisions[key][i, :] = np.array([np.nan for i in range(len(decisions[key][i, :]))])
            risk[key][i] = np.nan
            violation[key][i] = np.nan
                
    return (decisions, risk, risk, violation, X_test)

def evaluate_one_run_determ_constr(models, X, Y, X_est, Y_est, Nx_test, Ny_train, Ny_test, cond_mean, cond_std, 
    alpha = None, R = None, obj_coef = None, lb = -1, ub = 1,  sum_bound = 1, if_stoch_constr = False, verbose = False,
    generate_Y = None):
    p = X_est.shape[1]
    L = Y_est.shape[1]

    solver = partial(solve_mean_variance, R = R, alpha = alpha, obj_coef = obj_coef, lb = lb, ub = ub, sum_bound = sum_bound, if_stoch_constr = if_stoch_constr)
    # evaluate_risk = partial(evaluate_risk_singleX, alpha = alpha, obj_coef = obj_coef)
    evluate_violation = partial(evaluate_constr_violation, R = R)

    X_test = np.random.normal(size = (Nx_test, p))

    decisions = {}
    risk_comb = {}
    violation = {}
    risk = {}

    for key in models.keys():
        decisions[key] = np.zeros((Nx_test, L))
        risk_comb[key] = np.zeros(Nx_test)
        risk[key] = np.zeros(Nx_test)
        violation[key] = np.zeros(Nx_test)

    decisions["oracle"] = np.zeros((Nx_test, L))
    risk_comb["oracle"] = np.zeros(Nx_test)
    violation["oracle"] = np.zeros(Nx_test)
    risk["oracle"] = np.zeros(Nx_test)

    for i in range(Nx_test):
        if i % 100 == 0:
            print("the", i, "th evaluation sample.")
        Y_train = generate_Y(X_test[i, :][np.newaxis, :], cond_mean, cond_std, Ny = Ny_train)
        Y_test = generate_Y(X_test[i, :][np.newaxis, :], cond_mean, cond_std, Ny = Ny_test)

        for key in models.keys():
            weights = models[key].get_weights(X_test[i, :]) 
            try:
                (decision_temp, _, _, _) = solver(Y_est, weights = weights, if_weight = True)
                decisions[key][i, :] = decision_temp[:-1]
                risk_comb[key][i] = evaluate_risk_singleX(decisions[key][i, :], Y_test, obj_coef = obj_coef)
                risk[key][i] = evaluate_risk_singleX(decisions[key][i, :], Y_test,  obj_coef = 0)
                violation[key][i] = evluate_violation(decisions[key][i, :], Y_test)
            except:
                decisions[key][i, :] = np.array([np.nan for i in range(len(decisions[key][i, :]))])
                risk_comb[key][i] = np.nan
                risk[key][i] = np.nan
                violation[key][i] = np.nan

        key = "oracle"
        try:
            (decision_temp, _, _, _) = solver(Y_train)
            decisions[key][i, :] = decision_temp[:-1]
            risk_comb[key][i] = evaluate_risk_singleX(decisions[key][i, :], Y_test, obj_coef = obj_coef)
            risk[key][i] = evaluate_risk_singleX(decisions[key][i, :], Y_test, obj_coef = 0)
            violation[key][i] = evluate_violation(decisions[key][i, :], Y_test)
        except:
            decisions[key][i, :] = np.array([np.nan for i in range(len(decisions[key][i, :]))])
            risk_comb[key][i] = np.nan
            risk[key][i] = np.nan
            violation[key][i] = np.nan
                
    return (decisions, risk_comb, risk, violation, X_test)

def evaluate_risk_singleX(dec, Y_test, obj_coef = None):
    mu = np.mean(Y_test, 0)
    sigma = np.cov(Y_test, rowvar = False)
    return np.dot(np.dot(sigma, dec), dec) - obj_coef * np.dot(mu, dec)

def evaluate_constr_violation(dec, Y_test, R = None):
    return_temp = np.matmul(Y_test, dec)
    return R - return_temp.mean()

def extract_risk(results_eval):
    # results_eval: length is # of runs 
    # results_eval[0][0]: a list of decisions,
    # results_eval[0][1]: a list of risk,
    # results_eval[0][2]: a list of violation,

    exmp_key = list(results_eval[0][0])[0]
    N_test = results_eval[0][0][exmp_key].shape[0]
    runs = len(results_eval)
    loss = {key: np.zeros((N_test, runs)) for key in results_eval[0][1].keys()}

    for run in range(runs):
        (_, temp_loss, _, _, _) = results_eval[run]
        for key in loss.keys():
            loss[key][:, run] = temp_loss[key]

    risk = {}
    for key in loss.keys():
        risk[key] = np.nanmean(loss[key], axis = 0)
        
    return risk

def extract_rel_risk(risks, results_fit):

    rel_risks = {key: np.zeros(len(results_fit)) for key in results_fit[0].keys()}

    for key in results_fit[0].keys():
        rel_risks[key] = risks[key]/risks[str(key) + "_oracle"]

    return rel_risks

def evaluate_feature_split_freq(results_fit, p):

    feature_split_freq = {}
    for key in results_fit[0].keys():
        if str(key).find('rf') != -1:
            feature_split_freq[key] = np.zeros((len(results_fit), p))
            for run in range(len(results_fit)):
                feature_split_freq[key][run, :] = results_fit[run][key].compute_feature_split_freq(p)
    return feature_split_freq

def evaluate_opt_error_freq(results_fit):
    opt_error_freq = {}
    for key in results_fit[0].keys():
        if str(key).find('rf') != -1:
            opt_error_freq[key] = np.zeros(len(results_fit))
            for run in range(len(results_fit)):
                opt_error_freq[key][run] = results_fit[run][key].compute_opt_error_freq()

    return opt_error_freq

def evaluate_eval_opt_error(results_eval):
    eval_opt_error = {}
    for key in results_eval[0][2].keys():
        eval_opt_error[key] = np.zeros(len(results_eval))
        for run in range(len(results_eval)):
            eval_opt_error[key][run] =  np.isnan(results_eval[run][2][key]).mean()  
    return eval_opt_error

def evaluate_cond_violation(results_eval):
    exmp_key = list(results_eval[0][0])[0]
    N_test = results_eval[0][0][exmp_key].shape[0]
    runs = len(results_eval)
    violation = {key: np.zeros((N_test, runs)) for key in results_eval[0][1].keys()}

    for run in range(runs):
        (_, _, _, temp_violation, _) = results_eval[run]
        for key in violation.keys():
            violation[key][:, run] = np.maximum(temp_violation[key], 0)

    avg_violation = {}
    for key in violation.keys():
        avg_violation[key] = np.nanmean(violation[key], axis = 0)
        
    return avg_violation

def evaluate_mean_violation(results_eval):
    exmp_key = list(results_eval[0][0])[0]
    N_test = results_eval[0][0][exmp_key].shape[0]
    runs = len(results_eval)
    violation = {key: np.zeros((N_test, runs)) for key in results_eval[0][1].keys()}

    for run in range(runs):
        (_, _, _, temp_violation, _) = results_eval[run]
        for key in violation.keys():
            violation[key][:, run] = temp_violation[key]

    avg_violation = {}
    for key in violation.keys():
        avg_violation[key] = np.maximum(np.nanmean(violation[key], axis = 0), 0)
        
    return avg_violation

