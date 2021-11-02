import numpy as np
import warnings
from scipy.stats import gaussian_kde
import time
from joblib import Parallel, delayed
from functools import partial
from functools import reduce
from statsmodels.nonparametric.kernel_regression import KernelReg
import pickle 
from sklearn.preprocessing import OneHotEncoder 
from sklearn.model_selection import train_test_split, ShuffleSplit, StratifiedShuffleSplit
import scipy.stats 
from scipy.stats import truncnorm
import gurobipy as gp
from gurobipy import GRB
from sklearn.utils import check_random_state
import pandas as pd
from tree import *



import mkl
mkl.set_num_threads(1)


def solve_cvar(Y, A_mat = None, b_vec = None, alpha = None, verbose = False, weights = None, if_weight = False):

    L = Y.shape[1]
    n = Y.shape[0]
    if not if_weight:
        weights = np.ones(n)/n

    m = gp.Model()
    m.setParam("Threads", 1)
    z = pd.Series(m.addVars(L, lb = 0, name = 'z'), index = range(L))
    w = m.addVar(lb = 0, name = 'auxiliary')
    u = pd.Series(m.addVars(n, lb = 0, name = 'u'), index = range(n))
    m.update()

    risk = 1/(1 - alpha) * (weights * u).sum() + w 
    m.setObjective(risk, GRB.MINIMIZE)

    # piecewise linear 
    max_constraints = []
    for i in range(n):
        max_constraints.append(m.addConstr(u[i] >= Y[i, :].dot(z) - w))

    # Flow constraints 
    LP_constraints = []
    for i in range(A_mat.shape[0]):
        LP_constraints.append(m.addConstr(A_mat[i, :].dot(z) == b_vec[i]))

    m.update()

    if not verbose:
        m.setParam('OutputFlag', 0)

    m.optimize()

    try:
        decision = np.zeros(len(z)+1)
        decision[:-1] = np.array([zz.X for zz in z])
        decision[-1] = np.array(w.X)
        lam_de = [LP_constraints[i].Pi for i in range(len(LP_constraints))]
        lam_st = 0
        obj = m.objVal

    except:
        if verbose:
            print("optimization error!")
        decision = None
        lam_de = None
        lam_st = None
        obj = np.inf

    return (decision, lam_de, lam_st, obj)

def solve_simple(y, A_mat = None, b_vec = None, verbose = False):

    L = y.shape[0]

    m = gp.Model()
    m.setParam("Threads", 1)
    z = pd.Series(m.addVars(L, lb = 0, name = 'z'), index = range(L))
    m.update()
    risk = y.dot(z)
    m.setObjective(risk, GRB.MINIMIZE)


    # Flow constraints 
    LP_constraints = []
    for i in range(A_mat.shape[0]):
        LP_constraints.append(m.addConstr(A_mat[i, :].dot(z) == b_vec[i]))

    m.update()

    if not verbose:
        m.setParam('OutputFlag', 0)

    m.optimize()

    try:
        decision = np.array([zz.X for zz in z])
        lam_de = [LP_constraints[i].Pi for i in range(len(LP_constraints))]
        lam_st = 0
        obj = m.objVal

    except:
        if verbose:
            print("optimization error!")
        decision = None
        lam_de = None
        lam_st = None
        obj = np.inf

    return (decision, lam_de, lam_st, obj)
    
def search_active_constraint(node_Y, node_sol, A_mat = None, lb = 0, ub = 1, verbose = False):

    m = A_mat.shape[0]
    L = node_Y.shape[1]
    
    active_const_de = np.array([False]*(m + L))
    active_const_de[range(m)] = True   # all equality constraints are active 
    active_const_de[(m):(m+L)] = np.abs(node_sol[:-1] - lb) < 1e-6  # active positivity constraints 
    active_const_st = False

    return (active_const_de, active_const_st)

    
def compute_gradient(node_Y, node_sol, A_mat = None, alpha = None):

    z0 = node_sol[:-1]
    Y0 = np.matmul(node_Y, z0)
    q0 = np.quantile(Y0, alpha)

    upp = 1/(1 - alpha) * np.transpose(node_Y * (Y0 >= q0)[:, np.newaxis])
    low = -1/(1 - alpha) * (Y0 >= q0)[np.newaxis, :] + 1
    obj_grad = np.concatenate((upp, low), axis = 0)

    box_constr_gradient = np.concatenate((np.diag(np.ones(node_Y.shape[1])), np.zeros((1, node_Y.shape[1]))), axis= 0)

    # the last all-zero rows are for auxiliary variable w 
    constr_grad_de1 = np.concatenate((np.transpose(A_mat), np.zeros((1, A_mat.shape[0]))), axis = 0)
    constr_grad_de = np.concatenate((constr_grad_de1, -box_constr_gradient), axis = 1)
        # -box_constr_gradient is for positivity constraints z >= 0
    constr_grad_st = None
    
    return (obj_grad, constr_grad_de, constr_grad_st)


def compute_hessian(node_Y, node_sol, alpha = None):

    z0 = node_sol[:-1]
    Y0 = np.matmul(node_Y, z0)
    m0 = node_Y.mean(0)
    sigma0 = np.cov(node_Y, rowvar = False)
    q0 = np.quantile(Y0, alpha)

    EY_cond = m0 + np.matmul(sigma0, z0) / (np.matmul(np.transpose(z0), sigma0).dot(z0)) * (q0 - np.matmul(m0, z0))
    VY_cond = sigma0 - np.matmul(np.matmul(sigma0, z0)[:, np.newaxis], np.matmul(sigma0, z0)[np.newaxis, :])/(np.matmul(np.transpose(z0), sigma0).dot(z0))
    EYY_cond = VY_cond + np.matmul(EY_cond[:, np.newaxis], EY_cond[np.newaxis, :])
    mu0 = gaussian_kde(Y0)(q0)

    upp = np.concatenate((EYY_cond, -EY_cond[:, np.newaxis]), axis = 1)
    low = np.concatenate((-EY_cond[np.newaxis, :], np.ones((1, 1))), axis = 1)
    
    return mu0/(1-alpha) * np.concatenate((upp, low), axis = 0)
    
def compute_update_step(node_Y, node_sol, nu0, lambda0, 
                                      node_hessian, 
                                      node_obj_gradient, node_constr_gradient_de, node_constr_gradient_st,
                                      active_const_de, active_const_st,
                                      valid_side, 
                                      constraint = True):
    
    def compute_full_lhs(node_hessian, node_constr_gradient_st, node_constr_gradient_de):
        L22 = node_hessian
        H_grad = node_constr_gradient_de

        lhs_upper = np.concatenate((2*L22, H_grad), axis = 1)
        lhs_lower = np.concatenate((np.transpose(H_grad), np.zeros((H_grad.shape[1], H_grad.shape[1]))), axis = 1)
        lhs_full = np.concatenate((lhs_upper, lhs_lower), axis = 0)

        return lhs_full

    def compute_update(lhs, rhs):
        try:
            h = np.linalg.solve(lhs, rhs)[0:(len(node_sol)), :]
        except:
            lhs = lhs + np.diag([1e-3]*lhs.shape[1])
            h = np.linalg.solve(lhs, rhs)[0:(len(node_sol)), :]

        return h 

    def compute_child_gradient(node_obj_gradient, node_constr_gradient_st, split_obs):

        fj_grad = np.matmul(node_obj_gradient, split_obs)/split_obs.sum(0)
        gj_grad = None 

        return (fj_grad, gj_grad)

    def compute_full_rhs(node_Y, node_sol, f0_grad, g0_grad, 
                             node_obj_gradient, node_constr_gradient_st, 
                             nu0, lambda0, split_obs):
        (fj_grad, gj_grad) = compute_child_gradient(node_obj_gradient, node_constr_gradient_st, split_obs)

        rhs_upper = -2 * (fj_grad - f0_grad)
        rhs_lower = np.zeros((node_constr_gradient_de.shape[1], split_obs.shape[1]))
        rhs = np.concatenate((rhs_upper, rhs_lower), axis = 0)

        return (rhs, fj_grad, gj_grad)

    def compute_update_step_constr(lhs, node_Y, node_sol, f0_grad, g0_grad, 
                             node_obj_gradient, node_constr_gradient_st, 
                             nu0, lambda0, split_obs, index):
        (rhs, fj_grad, gj_grad) = compute_full_rhs(node_Y, node_sol, f0_grad, g0_grad, 
                         node_obj_gradient, node_constr_gradient_st, 
                         nu0, lambda0, split_obs)
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
        index = [True] * node_hessian.shape[0]  + list(active_const_de)
        
        lhs_full = compute_full_lhs(node_hessian, node_constr_gradient_st, node_constr_gradient_de)
        lhs = lhs_full[:, index]
        lhs = lhs[index, :]
        
        f0_grad = node_obj_gradient.mean(1)[:, np.newaxis]
        g0_grad = None 
        
        (node_h_left, fj_grad_left, gj_grad_left) = compute_update_step_constr(lhs, node_Y, node_sol, f0_grad, g0_grad, 
                         node_obj_gradient, node_constr_gradient_st, 
                         nu0, lambda0, valid_side, index)
        (node_h_right, fj_grad_right, gj_grad_right) = compute_update_step_constr(lhs, node_Y, node_sol, f0_grad, g0_grad, 
                         node_obj_gradient, node_constr_gradient_st, 
                         nu0, lambda0, ~valid_side, index)
        f_grad = {'0': f0_grad, 'j_left': fj_grad_left, 'j_right': fj_grad_right}
        g_grad = {'0': g0_grad, 'j_left': gj_grad_left, 'j_right': gj_grad_right}

    else:
        f0_grad = node_obj_gradient.mean(1)[:, np.newaxis]
        g0_grad = None

        (node_h_left, fj_grad_left, gj_grad_left) = compute_update_step_unconstr(node_obj_gradient, node_constr_gradient_st, node_hessian, valid_side)
        (node_h_right, fj_grad_right, gj_grad_right) = compute_update_step_unconstr(node_obj_gradient, node_constr_gradient_st, node_hessian, ~valid_side)
        f_grad = {'0': f0_grad, 'j_left': fj_grad_left, 'j_right': fj_grad_right}
        g_grad = {'0': g0_grad, 'j_left': gj_grad_left, 'j_right': gj_grad_right}

    return (node_h_left, node_h_right, f_grad, g_grad)

def compute_crit_rf(node_Y, node_sol, node_h_left, node_h_right, f_grad, g_grad, node_hessian, nu0, lambda0, valid_side, alpha = None):

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

def compute_crit_grf(node_Y, node_sol, node_h_left, node_h_right, f_grad, g_grad, node_hessian, nu0, lambda0, valid_side, alpha = None):
    # note that node_h_left and node_h_right are already average within the child nodes,
    #     so we first multiply the sample sizes in the child nodes to recover the sum 
    # then (rho_sum_left ** 2).sum(0) and (rho_sum_right ** 2).sum(0) compute the squared norm
    #     of the sum gradient 
    def compute_crit_grf_sub(node_h, split_obs):
        temp_n = split_obs.sum(0)
        rho_sum = temp_n[np.newaxis, :] * node_h
        crit_temp = (rho_sum ** 2).sum(0)/temp_n

        return crit_temp

    crit_left = compute_crit_grf_sub(node_h_left, valid_side)
    crit_right = compute_crit_grf_sub(node_h_right, ~valid_side)

    return (-crit_left - crit_right, None, None)


def compute_crit_approx_risk(node_Y, node_sol, node_h_left, node_h_right, f_grad, g_grad, node_hessian, nu0, lambda0, valid_side, alpha = None):

    def compute_risk_approx_risk(node_Y, node_h, f0_grad, fj_grad, g0_grad, gj_grad, node_hessian, nu0, lambda0):
        term1 = np.diagonal(np.matmul(np.matmul(np.transpose(node_h), node_hessian), node_h))
        term2 = 2 * np.diag(np.matmul(np.transpose(node_h), fj_grad - f0_grad))

        return (term1 + term2) 

    crit_left = compute_risk_approx_risk(node_Y, node_h_left, f_grad['0'], f_grad['j_left'], g_grad['0'], g_grad['j_left'], node_hessian, nu0, lambda0)
    crit_right = compute_risk_approx_risk(node_Y, node_h_right, f_grad['0'], f_grad['j_right'], g_grad['0'], g_grad['j_right'], node_hessian, nu0, lambda0)
    
    crit = valid_side.sum(0)/node_Y.shape[0] * crit_left + (1 - valid_side).sum(0)/node_Y.shape[0] * crit_right
    
    return (crit, crit_left, crit_right)

def compute_crit_approx_sol(node_Y, node_sol, node_h_left, node_h_right, f_grad, g_grad, node_hessian, nu0, lambda0, valid_side, alpha = None):
    
    def compute_risk_approx_sol(node_Y, node_sol, node_h, split_obs):
        sol_temp = node_sol[:, np.newaxis] + node_h
        w_temp = sol_temp[-1, :]
        z_temp = sol_temp[:-1, :]
        temp = np.maximum(np.matmul(node_Y, z_temp) - w_temp[np.newaxis, :], 0)
        obj_temp = 1/(1-alpha) * np.diag(np.matmul(np.transpose(split_obs), temp))/split_obs.sum(0) + w_temp 
        
        return obj_temp

    crit_left = compute_risk_approx_sol(node_Y, node_sol, node_h_left, valid_side)
    crit_right = compute_risk_approx_sol(node_Y, node_sol, node_h_right, ~valid_side)

    crit = valid_side.sum(0)/node_Y.shape[0] * crit_left + (1 - valid_side).sum(0)/node_Y.shape[0] * crit_right
    
    return (crit, crit_left, crit_right)

def compute_crit_approx_sol_proj_full(node_Y, node_sol, node_h_left, node_h_right, f_grad, g_grad, node_hessian, nu0, lambda0, valid_side, alpha = None, A_mat = None, b_vec = None):
    
    def compute_risk_approx_sol(node_Y, node_sol, node_h, split_obs):
        sol_temp = node_sol[:, np.newaxis] + node_h
        w_temp = sol_temp[-1, :]

        zero_ent = np.eye(node_sol[:-1].shape[0])[np.abs(node_sol[:-1]) < 1e-3]
        comb_mat = np.vstack((A_mat, zero_ent))
        comb_b = np.vstack((b_vec, np.zeros((zero_ent.shape[0], 1))))
        z_temp = sol_temp[:-1, :] + np.matmul(np.linalg.pinv(comb_mat), comb_b - comb_mat @ sol_temp[:-1, :])

        temp = np.maximum(np.matmul(node_Y, z_temp) - w_temp[np.newaxis, :], 0)
        obj_temp = 1/(1-alpha) * np.diag(np.matmul(np.transpose(split_obs), temp))/split_obs.sum(0) + w_temp 
        
        return obj_temp

    crit_left = compute_risk_approx_sol(node_Y, node_sol, node_h_left, valid_side)
    crit_right = compute_risk_approx_sol(node_Y, node_sol, node_h_right, ~valid_side)

    crit = valid_side.sum(0)/node_Y.shape[0] * crit_left + (1 - valid_side).sum(0)/node_Y.shape[0] * crit_right
    
    return (crit, crit_left, crit_right)


def compute_crit_random(node_Y, node_sol, node_h_left, node_h_right, f_grad, g_grad, node_hessian, nu0, lambda0, valid_side, alpha = None):
    return (np.random.rand(valid_side.shape[1]), None, None)

def impurity_rf(node_Y, node_sol, node_Y_left, node_Y_right, 
                         node_h_left, node_h_right, 
                         f_grad, g_grad, node_hessian, nu0, lambda0, best_split_ind, alpha = None):

    def compute_mse(Ytemp):
        temp = np.power(Ytemp - Ytemp.mean(0)[np.newaxis, :], 2)
        return temp.sum()
    
    imp_left = compute_mse(node_Y_left)
    imp_right = compute_mse(node_Y_right)
    imp_parent = compute_mse(node_Y)
    imp_dec = imp_parent - (imp_left + imp_right)
    
    return (imp_parent, imp_left, imp_right, imp_dec)

def impurity_approx_sol(node_Y, node_sol, node_Y_left, node_Y_right, 
                         node_h_left, node_h_right, 
                         f_grad, g_grad, node_hessian, nu0, lambda0, best_split_ind, alpha = None):
    
    node_h_left_final = node_h_left[:, best_split_ind]
    node_h_right_final = node_h_right[:, best_split_ind]
    fj_grad_left = f_grad['j_left'][:, best_split_ind][:, np.newaxis]
    fj_grad_right = f_grad['j_right'][:, best_split_ind][:, np.newaxis]
    f0_grad = f_grad['0']

    def compute_obj(decision, Ytemp):
        temp = np.maximum(np.matmul(Ytemp, decision[:-1])-decision[-1], 0)
        risk = 1/(1-alpha) * temp.mean(0) + decision[-1]
        return risk

    imp_left = compute_obj(node_sol+node_h_left_final, node_Y_left)*node_Y_left.shape[0]
    imp_right = compute_obj(node_sol+node_h_right_final, node_Y_right)*node_Y_right.shape[0]
    imp_parent = compute_obj(node_sol, node_Y)*node_Y.shape[0]
    imp_dec = imp_parent - (imp_left + imp_right)

    return (imp_parent, imp_left, imp_right, imp_dec)

def impurity_approx_risk(node_Y, node_sol, node_Y_left, node_Y_right, 
                         node_h_left, node_h_right, 
                         f_grad, g_grad, node_hessian, nu0, lambda0, best_split_ind, alpha = None):
    
    node_h_left_final = node_h_left[:, best_split_ind]
    node_h_right_final = node_h_right[:, best_split_ind]
    fj_grad_left = f_grad['j_left'][:, best_split_ind][:, np.newaxis]
    fj_grad_right = f_grad['j_right'][:, best_split_ind][:, np.newaxis]
    f0_grad = f_grad['0']

    def approx_risk_sub(Ytemp, node_sol, node_h_temp, fj_grad, f0_grad):
        fj = compute_obj(node_sol, Ytemp)
        term1 = 1/2*np.matmul(np.matmul(np.transpose(node_h_temp), node_hessian), node_h_temp)
        term2 = np.matmul(node_h_temp, fj_grad - f0_grad)
        return (fj + term1 + term2)*Ytemp.shape[0]

    def compute_obj(decision, Ytemp):
        temp = np.maximum(np.matmul(Ytemp, decision[:-1])-decision[-1], 0)
        risk = 1/(1-alpha) * temp.mean(0) + decision[-1]
        return risk

    imp_left = approx_risk_sub(node_Y_left, node_sol, node_h_left_final, fj_grad_left, f0_grad)
    imp_right = approx_risk_sub(node_Y_right, node_sol, node_h_right_final, fj_grad_right, f0_grad)
    imp_parent = compute_obj(node_sol, node_Y)*node_Y.shape[0]
    imp_dec = imp_parent - (imp_left + imp_right)
    
    return (imp_parent, imp_left[0], imp_right[0], imp_dec[0])


def evaluate_one_run(models, X, Y, X_est, Y_est, X_test, Y_test, A_mat = None, b_vec = None, alpha = None):
    Nx_test = X_test.shape[0]

    opt_solver = partial(solve_cvar, A_mat = A_mat, b_vec = b_vec, alpha = alpha)

    p = X_test.shape[1]
    L = Y_test.shape[1]
    decisions = {}
    risk = {}

    for key in models.keys():
        decisions[key] = np.zeros((Nx_test, L))

    for i in range(Nx_test):
        for key in models.keys():
            weights = models[key].get_weights(X_test[i, :]) 
            (decision_temp, _, _, _)  = opt_solver(Y_est, weights = weights, if_weight = True)
            decisions[key][i, :] = decision_temp[:-1]

    for key in models.keys():
        risk[key] = evaluate_risk(decisions[key], Y_test, alpha = alpha)

    (decision_temp, _, _, _) = opt_solver(Y_est, if_weight = False)
    decision_temp = decision_temp[:-1]
    test_overall_outcome = np.matmul(Y_test, decision_temp)
    test_w = np.quantile(test_overall_outcome, q = alpha)
    risk["vanilla"] = np.mean(1/(1-alpha) * np.maximum(test_overall_outcome - test_w, 0) + test_w, 0) 

    decision_oracle = compute_decisions_oracle(Y_test, A_mat = A_mat, b_vec = b_vec, alpha = alpha)
    risk["oracle"] = evaluate_risk(decision_oracle, Y_test, alpha = alpha)
    
    return (decisions, risk)

def compute_decisions_oracle(Y_test, A_mat = None, b_vec = None, alpha = 0.9):
    decisions = np.zeros((Y_test.shape[0], Y_test.shape[1]))
    
    for i in range(Y_test.shape[0]):
        y = Y_test[i, :]
        (decision_temp, _, _, _) = solve_simple(y, A_mat = A_mat, b_vec = b_vec, verbose = False)
        decisions[i, :] = decision_temp
        
    return decisions

def evaluate_risk(decisions, Y_test, alpha = 0.9):
    test_overall_outcome = np.diag(np.matmul(Y_test, np.transpose(decisions)))
    test_w = np.quantile(test_overall_outcome, q = alpha)
    risk = np.mean(1/(1-alpha) * np.maximum(test_overall_outcome - test_w, 0) + test_w, 0) 
    
    return risk 

def experiment_downtown_years(Y_train, X_train, Y_test, X_test, 
        A_mat = None, b_vec = None, alpha = 0.9, ub = 1, lb = 0, 
        subsample_ratio = 1, bootstrap = True, n_trees = 100, honesty = False, mtry = None, 
        min_leaf_size = None, max_depth = None, n_proposals = 100, balancedness_tol = None, verbose = False, seed = None):

    opt_solver = partial(solve_cvar, A_mat = A_mat, b_vec = b_vec, alpha = alpha)
    hessian_computer = partial(compute_hessian, alpha = alpha)
    active_constraint = partial(search_active_constraint,  A_mat = A_mat, lb = lb, ub = ub)
    gradient_computer = partial(compute_gradient,  A_mat = A_mat, alpha = alpha)

    models = {}
    times = {}

    time0 = time.time()
    forest_temp = forest(opt_solver = opt_solver, hessian_computer = hessian_computer,
                             gradient_computer = gradient_computer, 
                              search_active_constraint = active_constraint,
                             compute_update_step = compute_update_step,
                             crit_computer = partial(compute_crit_rf, alpha = alpha), 
                             impurity_computer = partial(impurity_rf, alpha = alpha), 
                            subsample_ratio=subsample_ratio, bootstrap = bootstrap, n_trees = n_trees, 
                             honesty = honesty, mtry = mtry,
                             min_leaf_size = min_leaf_size, max_depth = max_depth, 
                             n_proposals = n_proposals, balancedness_tol = balancedness_tol,
                             verbose = verbose, seed  = seed)
    forest_temp.fit(Y_train, X_train, Y_train, X_train)
    models['rf_rf'] = forest_temp
    times['rf_rf'] = time.time() - time0

    time0 = time.time()
    forest_temp = forest(opt_solver = opt_solver, hessian_computer = hessian_computer,
                             gradient_computer = gradient_computer, 
                              search_active_constraint = active_constraint,
                             compute_update_step = compute_update_step,
                             crit_computer = partial(compute_crit_random, alpha = alpha), 
                             impurity_computer = partial(impurity_rf, alpha = alpha), 
                            subsample_ratio=subsample_ratio, bootstrap = bootstrap, n_trees = n_trees, 
                             honesty = honesty, mtry = mtry,
                             min_leaf_size = min_leaf_size, max_depth = max_depth, 
                             n_proposals = n_proposals, balancedness_tol = balancedness_tol,
                             verbose = verbose, seed = seed)
    forest_temp.fit(Y_train, X_train, Y_train, X_train)
    models['rf_random'] = forest_temp
    times['rf_random'] = time.time() - time0

    time0 = time.time()
    forest_temp = forest(opt_solver = opt_solver, hessian_computer = hessian_computer,
                             gradient_computer = gradient_computer, 
                              search_active_constraint = active_constraint,
                             compute_update_step = partial(compute_update_step, constraint = False),
                             crit_computer = partial(compute_crit_grf, alpha = alpha), 
                             impurity_computer = partial(impurity_rf, alpha = alpha), 
                            subsample_ratio=subsample_ratio, bootstrap = bootstrap, n_trees = n_trees, 
                             honesty = honesty, mtry = mtry,
                             min_leaf_size = min_leaf_size, max_depth = max_depth, 
                             n_proposals = n_proposals, balancedness_tol = balancedness_tol,
                             verbose = verbose, seed = seed)
    forest_temp.fit(Y_train, X_train, Y_train, X_train)
    models['grf'] = forest_temp
    times['grf'] = time.time() - time0

    time0 = time.time()
    forest_temp = forest(opt_solver = opt_solver, hessian_computer = hessian_computer,
                             gradient_computer = gradient_computer, 
                              search_active_constraint = active_constraint,
                             compute_update_step = compute_update_step,
                             crit_computer = partial(compute_crit_approx_sol_proj_full, alpha = alpha, A_mat = A_mat, b_vec = b_vec), 
                             impurity_computer = partial(impurity_approx_sol, alpha = alpha), 
                            subsample_ratio=subsample_ratio, bootstrap = bootstrap, n_trees = n_trees, 
                             honesty = honesty, mtry = mtry,
                             min_leaf_size = min_leaf_size, max_depth = max_depth, 
                             n_proposals = n_proposals, balancedness_tol = balancedness_tol,
                             verbose = verbose, seed = seed)
    forest_temp.fit(Y_train, X_train, Y_train, X_train)
    models['rf_approx_sol'] = forest_temp
    times['rf_approx_sol'] = time.time() - time0

    time0 = time.time()
    forest_temp = forest(opt_solver = opt_solver, hessian_computer = hessian_computer,
                             gradient_computer = gradient_computer, 
                              search_active_constraint = active_constraint,
                             compute_update_step = compute_update_step,
                             crit_computer = partial(compute_crit_approx_risk, alpha = alpha),
                             impurity_computer = partial(impurity_approx_risk, alpha = alpha),  
                            subsample_ratio=subsample_ratio, bootstrap = bootstrap, n_trees = n_trees, 
                             honesty = honesty, mtry = mtry,
                             min_leaf_size = min_leaf_size, max_depth = max_depth, 
                             n_proposals = n_proposals, balancedness_tol = balancedness_tol,
                             verbose = verbose, seed = seed)
    forest_temp.fit(Y_train, X_train, Y_train, X_train)
    models['rf_approx_risk'] = forest_temp
    times['rf_approx_risk'] = time.time() - time0

    time0 = time.time()
    forest_temp = forest(opt_solver = opt_solver, hessian_computer = hessian_computer,
                             gradient_computer = gradient_computer, 
                              search_active_constraint = active_constraint,
                             compute_update_step = partial(compute_update_step, constraint = False),
                             crit_computer = partial(compute_crit_approx_sol, alpha = alpha), 
                             impurity_computer = partial(impurity_approx_sol, alpha = alpha), 
                            subsample_ratio=subsample_ratio, bootstrap = bootstrap, n_trees = n_trees, 
                             honesty = honesty, mtry = mtry,
                             min_leaf_size = min_leaf_size, max_depth = max_depth, 
                             n_proposals = n_proposals, balancedness_tol = balancedness_tol,
                             verbose = verbose, seed = seed)
    forest_temp.fit(Y_train, X_train, Y_train, X_train)
    models['rf_approx_sol_unconstr'] = forest_temp
    times['rf_approx_sol_unconstr'] = time.time() - time0

    time0 = time.time()
    forest_temp = forest(opt_solver = opt_solver, hessian_computer = hessian_computer,
                             gradient_computer = gradient_computer, 
                              search_active_constraint = active_constraint,
                             compute_update_step = partial(compute_update_step, constraint = False),
                             crit_computer = partial(compute_crit_approx_risk, alpha = alpha),
                             impurity_computer = partial(impurity_approx_risk, alpha = alpha),  
                            subsample_ratio=subsample_ratio, bootstrap = bootstrap, n_trees = n_trees, 
                             honesty = honesty, mtry = mtry,
                             min_leaf_size = min_leaf_size, max_depth = max_depth, 
                             n_proposals = n_proposals, balancedness_tol = balancedness_tol,
                             verbose = verbose, seed = seed)
    forest_temp.fit(Y_train, X_train, Y_train, X_train)
    models['rf_approx_risk_unconstr'] = forest_temp
    times['rf_approx_risk_unconstr'] = time.time() - time0

    return (models, times)

def evaluate_feature_split_freq(results_fit, p):

    feature_split_freq = {}
    for key in results_fit[0].keys():
        if str(key).find('rf') != -1:
            feature_split_freq[key] = np.zeros((len(results_fit), p))
            for i in range(len(results_fit)):
                feature_split_freq[key][i, :] = results_fit[i][key].compute_feature_split_freq(p)
    return feature_split_freq

def evaluate_feature_importance(results_fit, p):

    feature_importance = {}
    for key in results_fit[0].keys():
        if str(key).find('rf') != -1:
            feature_importance[key] = np.zeros((len(results_fit), p))
            for i in range(len(results_fit)):
                try:
                    feature_importance[key][i, :] = results_fit[i][key].compute_impurity_fi(p)
                except:
                    continue 

    feature_importance_opt = {}
    for key in results_fit[0].keys():
        if str(key).find('rf') != -1:
            feature_importance_opt[key] = np.zeros((len(results_fit), p))
            for i in range(len(results_fit)):
                try:
                    feature_importance_opt[key][i, :] = results_fit[i][key].compute_impurity_opt_fi(p)
                except:
                    continue

    return (feature_importance, feature_importance_opt)