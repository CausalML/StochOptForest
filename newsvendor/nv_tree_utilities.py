import numpy as np
import warnings
from scipy.stats import gaussian_kde
import time
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
from sklearn.neighbors import NearestNeighbors
from tree import *

import mkl
mkl.set_num_threads(1)

def get_truncated_normal(mean=0, sd=1, low=0, upp=10, seed = None):
    if seed != None:
        np.random.seed(seed)
    return truncnorm(
            (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def generate_Y(X, cond_mean, cond_std, Ny = None, seed = None):
    if seed != None:
        np.random.seed(seed)

    L = len(cond_mean)

    if Ny is None:
        Ny = X.shape[0]

    Y = np.zeros((Ny, L))
    for l in range(L):
        if Ny > X.shape[0]:
            X_temp = np.repeat(X, Ny, axis = 0)
            Y[:, l] = get_truncated_normal(low = 0, upp = 100, mean = cond_mean[l](X_temp), sd = cond_std[l](X_temp), seed = seed).rvs()
        else:
            Y[:, l] = get_truncated_normal(low = 0, upp = 100, mean = cond_mean[l](X), sd = cond_std[l](X), seed = seed).rvs()

    return Y

def weighted_quantile(Y, w, levels = None):
    # assert: 
    assert Y.shape[1] == len(levels)
    w = w/w.sum()
    
    quantiles = np.zeros(len(levels))
    
    for l in range(Y.shape[1]):
        Y_l = Y[:, l]
    
        ix = np.argsort(Y_l)
        Y_l = Y_l[ix] # sort Y
        w = w[ix] # sort weights
        cdf = np.cumsum(w)  # weighted CDF function
        
        if levels[l] < 0:
            quantiles[l] = 0
        else:
            quantiles[l] = np.interp(levels[l], cdf, Y_l)
    # np.interp: interpolate the Y value of level according to its position in cdf 
    return quantiles

def compute_quantile_general_level(Y, levels):
    # function that computes quantiles with negative levels --> return 0 for negative quantiles
    #    len(levels) should be equal to Y.shape[1]
    #    we compute the quantile of Y[:, i] with level levels[i]
    
    # we only compute quantile for positive levels, and opt solutions for negative levels are set to 0
    sol_temp = np.zeros(len(levels))
    q_temp = np.quantile(Y[:, levels > 0], q = levels[levels > 0], axis = 0) 
    # number of rows is equal to number of quantile levels that are positive
    # number of columns is equal to the number of column of Y, where we only need those with levels > 0
    # thus the desired quantiles are given by the diagonal entries of q_temp[:, temp_level > 0]
    sol_temp[levels > 0] = np.diagonal(q_temp)
    
    return sol_temp

def solve_multi_nv(Y, h_list = None, b_list = None, C = None, verbose = False, if_weight = False, weights = None):
    if verbose:
        print("h_list: ", h_list)
        print("b_list: ", b_list)
        print("constraint: ", C)
        
    def compute_obj(decision):
        obj = np.mean(np.matmul((decision[np.newaxis, :] - Y) * (decision - Y >= 0), h_list) + np.matmul((Y - decision[np.newaxis, :]) * (Y - decision > 0), b_list))
        return obj
        
    w = weights
    prop = 0.   # this is the lagrangian multiplier for the capacity constraint 

    # initial comparison: if the constraint can be satisfied with zero lagrangian multiplier, then 
    # return it directly 
    if if_weight:
        sol_temp = weighted_quantile(Y, w = w, levels = b_list/(b_list + h_list))
    else:
        sol_temp = compute_quantile_general_level(Y, b_list/(b_list + h_list))

    if sol_temp.sum() <= C:
        if verbose:
            print("capacity constraint not active!")
            print("solution: ", sol_temp)
            
        decision = np.round(sol_temp, 8)
        lam_de = 0
        lam_st = 0
        obj = compute_obj(decision)
        
        return (decision, lam_de, lam_st, obj)
    else: 
        if verbose:
            print("capacity constraint is infeasible!")
            print("sol_temp.sum() - C = :", sol_temp.sum() - C)

    # otherwise start bisection: 
    #    low --> constraint always infeasible
    #    high --> constraint always feasible 
    #    prop = (low + high)/2 --> the proposal lagrangian multiplier value for checking the constraint 
    #    temp_level: the quantile level corresponding to the proposal lagrangian multiplier  
    low = 0; high = max(b_list)

    i = 0
    delta = np.inf

    while not((delta >= -0.01 and delta <= 0) or (high - low <= 0.005 and delta <= 0)): 
        # exit the loop if we find approximate root: delta >= -0.01 and delta <= 0
        #     or if there is some discontinuity at one search point that prevents exact convergence:
        #           high - low <= 0.01 and delta <= 0
        i = i + 1
        if verbose:
            print("iteration:", i)
            print("low:", low)
            print("high:", high)

        prop = (low + high)/2

        temp_level = (b_list - prop)/(b_list + h_list)
        if verbose:
            print("proposal_level:", temp_level)

        if if_weight:
            sol_temp = weighted_quantile(Y, w = w, levels = temp_level)
        else:
            sol_temp = compute_quantile_general_level(Y, temp_level)

        if verbose:
            print("solution: ", sol_temp)
        
        if sol_temp.sum() <= C: # constraint feasible --> set to high 
            high = prop
        else: # constraint infeasible --> set to low 
            low = prop
        delta = sol_temp.sum() - C

        if verbose:
            print("sol_temp.sum() - C:", sol_temp.sum() - C)

    # if we encounter a discontinuous point lambda, then usually this lambda is in b_list
    #       Consider the item whose b = lambda. When lambda approaches b from left, the quantile for this item is 0,
    #       and the constraint may still not be satisfied, but when lambda approaches b from the right, 
    #       the quantile can be so large that the constraint is violated. Because the quantiles of such boundary items
    #       are discontinuous at lambda = b, while quantiles for other items are continuous, the constraint would also 
    #       be discontinuous in lambda: for lambda < b it is very inactive but for lambda > b it is violated by a lot
    #       thus the recursion above never stops, and lambda would gradually approach b. 
    #       our solution here is to set lambda still close to b, but we redistribute 
    #       so we evenly redistribute the constraint surplus C - sol_temp.sum() to the boundary items   
    # detecting the problem by problem_index
        # discont_index: items whose b levels are close to prop, which are those discontinuity problem can happen
        # zero_index: items where the decision is also close to 0
        # note that if the procedure converges and stops normally, then C - sol_temp.sum() is small so redistribution
        # doesn't matter much 
    discont_index = np.abs(prop - b_list) < 0.01 
    zero_index = sol_temp < 0.001  
    problem_index = discont_index * zero_index
    if (C - sol_temp.sum() > 0.01) and sum(problem_index) > 0:
        sol_temp[problem_index] = (C - sol_temp.sum())/sum(problem_index)

    decision = np.round(sol_temp, 8)
    lam_de = np.round(prop, 8)
    lam_st = 0
    obj = compute_obj(decision)

    return (decision, lam_de, lam_st, obj)


def search_active_constraint(node_Y, node_sol, C = None, verbose = None):

    active_const_de = np.array([False] * (len(node_sol) + 1))
    active_const_de[:len(node_sol)] = np.abs(node_sol) < 1e-4
    active_const_de[-1] = np.abs(node_sol.sum()-C) < 1e-4
        
    active_const_st = None 

    return (active_const_de, active_const_st)

def compute_gradient(node_Y, node_sol, h_list = None, b_list = None, C = None):
    obj_grad = np.matmul(node_Y <= node_sol[np.newaxis, :], np.diag(h_list + b_list)) - b_list[np.newaxis, :]
    constr_grad_de = np.concatenate((np.diag(-np.ones(len(node_sol))), -np.ones((len(node_sol), 1))), axis = 1)
    constr_grad_st = None 

    return (np.transpose(obj_grad), constr_grad_de, constr_grad_st)

def compute_hessian(node_Y, node_sol, h_list = None, b_list = None, C = None):
    temp = np.array([(h_list[l] + b_list[l]) * gaussian_kde(node_Y[:, l])(node_sol[l])[0] for l in range(len(node_sol))])

    return np.diag(temp)

def compute_update_step(node_Y, node_sol, nu0, lambda0, 
                                      node_hessian, 
                                      node_obj_gradient, node_constr_gradient_de, node_constr_gradient_st,
                                      active_const_de, active_const_st,
                                      valid_side, 
                                      constraint = True):
    
    def compute_child_gradient(node_obj_gradient, node_constr_gradient_st, split_obs):

        fj_grad = np.matmul(node_obj_gradient, split_obs)/split_obs.sum(0)
        gj_grad = None

        return (fj_grad, gj_grad)
    
    def compute_full_lhs(node_hessian, node_constr_gradient_st, node_constr_gradient_de):
        L22 = node_hessian
        H_grad = node_constr_gradient_de

        lhs_upper = np.concatenate((2*L22, H_grad), axis = 1)
        lhs_lower = np.concatenate((np.transpose(H_grad),  np.zeros((H_grad.shape[1], H_grad.shape[1]))), axis = 1)
        lhs_full = np.concatenate((lhs_upper, lhs_lower), axis = 0)

        return lhs_full
    
    def compute_full_rhs(node_Y, node_sol, f0_grad, g0_grad, 
                         node_obj_gradient, node_constr_gradient_st, 
                         nu0, lambda0, split_obs):
        (fj_grad, gj_grad) = compute_child_gradient(node_obj_gradient, node_constr_gradient_st, split_obs)

        rhs_upper = -2 * (fj_grad - f0_grad) 
        rhs_lower = np.zeros((node_constr_gradient_de.shape[1], split_obs.shape[1]))
        rhs = np.concatenate((rhs_upper, rhs_lower), axis = 0)

        return (rhs, fj_grad, gj_grad)
    
    def compute_update(lhs, rhs):
        try:
            h = np.linalg.solve(lhs, rhs)[0:(len(node_sol)), :]
        except:
            lhs = lhs + np.diag([0.05]*lhs.shape[1])
            h = np.linalg.solve(lhs, rhs)[0:(len(node_sol)), :]

        return h
    
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
        index = [True] * node_hessian.shape[0]  +  list(active_const_de)
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
        g_grad = {'0': None, 'j_left': None, 'j_right': None}
        
    else:
        f0_grad = node_obj_gradient.mean(1)[:, np.newaxis]
        g0_grad = None 

        (node_h_left, fj_grad_left, gj_grad_left) = compute_update_step_unconstr(node_obj_gradient, node_constr_gradient_st, node_hessian, valid_side)
        (node_h_right, fj_grad_right, gj_grad_right) = compute_update_step_unconstr(node_obj_gradient, node_constr_gradient_st, node_hessian, ~valid_side)
        f_grad = {'0': f0_grad, 'j_left': fj_grad_left, 'j_right': fj_grad_right}
        g_grad = {'0': None, 'j_left': None, 'j_right': None}
    
    return (node_h_left, node_h_right, f_grad, g_grad)



def compute_crit_grf(node_Y, node_sol, node_h_left, node_h_right, f_grad, g_grad, node_hessian, nu0, lambda0, valid_side):
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

def compute_crit_approx_risk(node_Y, node_sol, node_h_left, node_h_right, f_grad, g_grad, node_hessian, nu0, lambda0, valid_side):

    def compute_risk_approx_risk(node_Y, node_h, f0_grad, fj_grad, g0_grad, gj_grad, node_hessian, nu0, lambda0):
        term1 = np.diagonal(np.matmul(np.matmul(np.transpose(node_h), node_hessian), node_h))
        term2 = 2 * np.diag(np.matmul(np.transpose(node_h), fj_grad - f0_grad))

        return (term1 + term2) 

    crit_left = compute_risk_approx_risk(node_Y, node_h_left, f_grad['0'], f_grad['j_left'], g_grad['0'], g_grad['j_left'], node_hessian, nu0, lambda0)
    crit_right = compute_risk_approx_risk(node_Y, node_h_right, f_grad['0'], f_grad['j_right'], g_grad['0'], g_grad['j_right'], node_hessian, nu0, lambda0)
    
    crit = valid_side.sum(0)/node_Y.shape[0] * crit_left + (1 - valid_side).sum(0)/node_Y.shape[0] * crit_right
    
    return (crit, crit_left, crit_right)

def compute_crit_approx_sol(node_Y, node_sol, node_h_left, node_h_right, f_grad, g_grad, node_hessian, nu0, lambda0, valid_side, h_list = None, b_list = None):
    def compute_risk_approx_sol(node_Y, node_sol, node_h, split_obs):
        sol_temp = node_sol[:, np.newaxis] + node_h
        
        temp_n = split_obs.sum(0)
        temp_crit = np.zeros((node_Y.shape[1], split_obs.shape[1]))

        for l in range(node_Y.shape[1]):
            Yl = node_Y[:, l][:, np.newaxis] # demand corresponding to the item l 
            sol_l = sol_temp[l, :][np.newaxis, :]  # one-step update of decision for item l in left child node 

            temp_loss_l = h_list[l] * (sol_l - Yl) * (sol_l - Yl >= 0) + b_list[l] * (Yl - sol_l) * (sol_l - Yl < 0)
            temp_crit[l, :] = (temp_loss_l * split_obs).sum(0)/temp_n

        return temp_crit.sum(0)

    crit_left = compute_risk_approx_sol(node_Y, node_sol, node_h_left,  valid_side)
    crit_right = compute_risk_approx_sol(node_Y, node_sol, node_h_right, ~valid_side)

    crit = valid_side.sum(0)/node_Y.shape[0] * crit_left + (1 - valid_side).sum(0)/node_Y.shape[0] * crit_right

    return (crit, crit_left, crit_right)

def impurity_rf(node_Y, node_sol, node_Y_left, node_Y_right, 
                        node_h_left, node_h_right, f_grad, node_hessian, best_split_ind, h_list = None, b_list = None, C = None):
    
    def compute_mse(Ytemp):
        temp = np.power(Ytemp - Ytemp.mean(0)[np.newaxis, :], 2)
        return temp.sum()
    
    imp_left = compute_mse(node_Y_left)
    imp_right = compute_mse(node_Y_right)
    imp_parent = compute_mse(node_Y)
    imp_dec = imp_parent - (imp_left + imp_right)
    
    return (imp_parent, imp_left, imp_right, imp_dec)

def impurity_approx_risk(node_Y, node_sol, node_Y_left, node_Y_right, 
                        node_h_left, node_h_right, f_grad, node_hessian, best_split_ind, h_list = None, b_list = None, C = None):
    
    
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
        temp = decision[np.newaxis, :] - Ytemp
        risk = (np.matmul(temp * (temp >= 0), h_list) + np.matmul(-temp * (temp < 0), b_list)).mean(0)
        return risk

    imp_left = approx_risk_sub(node_Y_left, node_sol, node_h_left_final, fj_grad_left, f0_grad)
    imp_right = approx_risk_sub(node_Y_right, node_sol, node_h_right_final, fj_grad_right, f0_grad)
    imp_parent = compute_obj(node_sol, node_Y)*node_Y.shape[0]
    imp_dec = imp_parent - (imp_left + imp_right)
    
    return (imp_parent, imp_left[0], imp_right[0], imp_dec[0])

def impurity_approx_sol(node_Y, node_sol, node_Y_left, node_Y_right, 
                        node_h_left, node_h_right, f_grad, node_hessian, best_split_ind, h_list = None, b_list = None, C = None):
    
    node_h_left_final = node_h_left[:, best_split_ind]
    node_h_right_final = node_h_right[:, best_split_ind]
    fj_grad_left = f_grad['j_left'][:, best_split_ind][:, np.newaxis]
    fj_grad_right = f_grad['j_right'][:, best_split_ind][:, np.newaxis]
    f0_grad = f_grad['0']

    def compute_obj(decision, Ytemp):
        temp = decision[np.newaxis, :] - Ytemp
        risk = (np.matmul(temp * (temp >= 0), h_list) + np.matmul(-temp * (temp < 0), b_list)).mean(0)
        return risk

    imp_left = compute_obj(node_sol+node_h_left_final, node_Y_left)*node_Y_left.shape[0]
    imp_right = compute_obj(node_sol+node_h_right_final, node_Y_right)*node_Y_right.shape[0]
    imp_parent = compute_obj(node_sol, node_Y)*node_Y.shape[0]
    imp_dec = imp_parent - (imp_left + imp_right)
    
    return (imp_parent, imp_left, imp_right, imp_dec)

def impurity_oracle(node_Y, node_sol, node_Y_left, node_Y_right, 
                        node_h_left, node_h_right, f_grad, node_hessian, best_split_ind, h_list = None, b_list = None, C = None):
    
    def compute_obj(decision, Ytemp):
        temp = decision[np.newaxis, :] - Ytemp
        risk = (np.matmul(temp * (temp >= 0), h_list) + np.matmul(-temp * (temp < 0), b_list)).mean(0)
        return risk

    (_, _, _, obj) = solve_multi_nv(node_Y_left, h_list = h_list, b_list = b_list, C = C)
    imp_left = obj * node_Y_left.shape[0]
    (_, _, _, obj) = solve_multi_nv(node_Y_right, h_list = h_list, b_list = b_list, C = C)
    imp_right = obj * node_Y_right.shape[0]
    imp_parent = compute_obj(node_sol, node_Y)*node_Y.shape[0]
    imp_dec = imp_parent - (imp_left + imp_right)
    
    return (imp_parent, imp_left, imp_right, imp_dec)

def compare_forest_one_run_honesty(X, Y, X_est, Y_est, 
    h_list = None, b_list = None, C = None, 
    n_trees = 500, honesty = False, mtry = 5, verbose = False, subsample_ratio = 0.5,
    oracle = False,min_leaf_size=10, max_depth=100, n_proposals = 200, 
    balancedness_tol = 0.3, bootstrap = False, seed = None):

    models = {}

    opt_solver = partial(solve_multi_nv, h_list = h_list, b_list = b_list, C = C, verbose = verbose)
    hessian_computer = partial(compute_hessian, h_list = h_list, b_list = b_list, C = C)
    gradient_computer = partial(compute_gradient, h_list = h_list, b_list = b_list, C = C)
    active_constraint = partial(search_active_constraint, C = C)
    update_step = partial(compute_update_step, constraint = True)

    forest_temp = forest(opt_solver = opt_solver, hessian_computer = hessian_computer,
                         gradient_computer = gradient_computer, 
                          search_active_constraint = active_constraint,
                         compute_update_step = update_step,
                         crit_computer = partial(compute_crit_approx_sol, h_list =  h_list, b_list = b_list),
                         impurity_computer = partial(impurity_approx_sol, h_list = h_list, b_list = b_list, C = C), 
                         subsample_ratio=subsample_ratio, bootstrap = bootstrap, n_trees = n_trees, 
                         honesty = True, mtry = mtry,
                         min_leaf_size = min_leaf_size, max_depth = max_depth, 
                         n_proposals = n_proposals, balancedness_tol = balancedness_tol,
                         verbose = verbose, seed = seed)
    forest_temp.fit(Y, X, Y_est, X_est)
    models['rf_approx_sol_honest'] = forest_temp

    forest_temp = forest(opt_solver = opt_solver, hessian_computer = hessian_computer,
                         gradient_computer = gradient_computer, 
                          search_active_constraint = active_constraint,
                         compute_update_step = update_step,
                         crit_computer = compute_crit_approx_risk, 
                         impurity_computer = partial(impurity_approx_risk, h_list = h_list, b_list = b_list, C = C), 
                        subsample_ratio=subsample_ratio, bootstrap = bootstrap, n_trees = n_trees, 
                         honesty = True, mtry = mtry,
                         min_leaf_size = min_leaf_size, max_depth = max_depth, 
                         n_proposals = n_proposals, balancedness_tol = balancedness_tol,
                         verbose = verbose, seed = seed)
    forest_temp.fit(Y, X, Y_est, X_est)
    models['rf_approx_risk_honest'] = forest_temp

    forest_temp = forest(opt_solver = opt_solver, hessian_computer = hessian_computer,
                         gradient_computer = gradient_computer, 
                          search_active_constraint = active_constraint,
                         compute_update_step = update_step,
                         crit_computer = partial(compute_crit_approx_sol, h_list =  h_list, b_list = b_list),
                         impurity_computer = partial(impurity_approx_sol, h_list = h_list, b_list = b_list, C = C), 
                         subsample_ratio=subsample_ratio, bootstrap = bootstrap, n_trees = n_trees, 
                         honesty = False, mtry = mtry,
                         min_leaf_size = min_leaf_size, max_depth = max_depth, 
                         n_proposals = n_proposals, balancedness_tol = balancedness_tol,
                         verbose = verbose, seed = seed)
    forest_temp.fit(Y, X, Y_est, X_est)
    models['rf_approx_sol_dishonest'] = forest_temp

    forest_temp = forest(opt_solver = opt_solver, hessian_computer = hessian_computer,
                         gradient_computer = gradient_computer, 
                          search_active_constraint = active_constraint,
                         compute_update_step = update_step,
                         crit_computer = compute_crit_approx_risk, 
                         impurity_computer = partial(impurity_approx_risk, h_list = h_list, b_list = b_list, C = C), 
                        subsample_ratio=subsample_ratio, bootstrap = bootstrap, n_trees = n_trees, 
                         honesty = False, mtry = mtry,
                         min_leaf_size = min_leaf_size, max_depth = max_depth, 
                         n_proposals = n_proposals, balancedness_tol = balancedness_tol,
                         verbose = verbose, seed = seed)
    forest_temp.fit(Y, X, Y_est, X_est)
    models['rf_approx_risk_dishonest'] = forest_temp

    return models

def compare_forest_one_run(X, Y, X_est, Y_est, 
    h_list = None, b_list = None, C = None, 
    n_trees = 500, honesty = False, mtry = 5, verbose = False, subsample_ratio = 0.5,
    oracle = False,min_leaf_size=10, max_depth=100, n_proposals = 200, 
    balancedness_tol = 0.3, bootstrap = False, seed = None):
    models = {}

    opt_solver = partial(solve_multi_nv, h_list = h_list, b_list = b_list, C = C, verbose = verbose)
    hessian_computer = partial(compute_hessian, h_list = h_list, b_list = b_list, C = C)
    gradient_computer = partial(compute_gradient, h_list = h_list, b_list = b_list, C = C)
    active_constraint = partial(search_active_constraint, C = C)
    update_step = partial(compute_update_step, constraint = True)


    forest_temp = forest(opt_solver = opt_solver, hessian_computer = hessian_computer,
                         gradient_computer = gradient_computer, 
                          search_active_constraint = active_constraint,
                         compute_update_step = update_step,
                         crit_computer = partial(compute_crit_approx_sol, h_list =  h_list, b_list = b_list),
                         impurity_computer = partial(impurity_approx_sol, h_list = h_list, b_list = b_list, C = C), 
                         subsample_ratio=subsample_ratio, bootstrap = bootstrap, n_trees = n_trees, 
                         honesty = honesty, mtry = mtry,
                         min_leaf_size = min_leaf_size, max_depth = max_depth, 
                         n_proposals = n_proposals, balancedness_tol = balancedness_tol,
                         verbose = verbose, seed = seed)
    forest_temp.fit(Y, X, Y_est, X_est)
    models['rf_approx_sol'] = forest_temp

    forest_temp = forest(opt_solver = opt_solver, hessian_computer = hessian_computer,
                         gradient_computer = gradient_computer, 
                          search_active_constraint = active_constraint,
                         compute_update_step = update_step,
                         crit_computer = compute_crit_approx_risk, 
                         impurity_computer = partial(impurity_approx_risk, h_list = h_list, b_list = b_list, C = C), 
                        subsample_ratio=subsample_ratio, bootstrap = bootstrap, n_trees = n_trees, 
                         honesty = honesty, mtry = mtry,
                         min_leaf_size = min_leaf_size, max_depth = max_depth, 
                         n_proposals = n_proposals, balancedness_tol = balancedness_tol,
                         verbose = verbose, seed = seed)
    forest_temp.fit(Y, X, Y_est, X_est)
    models['rf_approx_risk'] = forest_temp

    forest_temp = forest(opt_solver = opt_solver, hessian_computer = hessian_computer,
                         gradient_computer = gradient_computer, 
                          search_active_constraint = active_constraint,
                         compute_update_step = update_step,
                         crit_computer = compute_crit_rf, 
                         impurity_computer = partial(impurity_rf, h_list = h_list, b_list = b_list, C = C), 
                        subsample_ratio=subsample_ratio, bootstrap = bootstrap, n_trees = n_trees, 
                         honesty = honesty, mtry = mtry,
                         min_leaf_size = min_leaf_size, max_depth = max_depth, 
                         n_proposals = n_proposals, balancedness_tol = balancedness_tol,
                         verbose = verbose, seed = seed)
    forest_temp.fit(Y, X, Y_est, X_est)
    models['rf_rf'] = forest_temp

    forest_temp = forest(opt_solver = opt_solver, hessian_computer = hessian_computer,
                         gradient_computer = gradient_computer, 
                          search_active_constraint = active_constraint,
                         compute_update_step = update_step,
                         crit_computer = compute_crit_grf, 
                         impurity_computer = partial(impurity_rf, h_list = h_list, b_list = b_list, C = C), 
                        subsample_ratio=subsample_ratio, bootstrap = bootstrap, n_trees = n_trees, 
                         honesty = honesty, mtry = mtry,
                         min_leaf_size = min_leaf_size, max_depth = max_depth, 
                         n_proposals = n_proposals, balancedness_tol = balancedness_tol,
                         verbose = verbose, seed = seed)
    forest_temp.fit(Y, X, Y_est, X_est)
    models['rf_grf'] = forest_temp

    if oracle:
        forest_temp = forest(opt_solver = opt_solver, hessian_computer = hessian_computer,
                         gradient_computer = gradient_computer, 
                          search_active_constraint = active_constraint,
                         compute_update_step = update_step,
                         crit_computer = partial(compute_crit_oracle, solver = opt_solver), 
                         impurity_computer = partial(impurity_oracle, h_list = h_list, b_list = b_list, C = C), 
                        subsample_ratio=subsample_ratio, bootstrap = bootstrap, n_trees = n_trees, 
                         honesty = honesty, mtry = mtry,
                         min_leaf_size = min_leaf_size, max_depth = max_depth, 
                         n_proposals = n_proposals, balancedness_tol = balancedness_tol,
                         verbose = verbose, seed = seed)
        forest_temp.fit(Y, X, Y_est, X_est)
        models['rf_oracle'] = forest_temp

    return models

def compare_adaptive_nonadaptive_one_run(X, Y, X_est, Y_est, 
    h_list = None, b_list = None, C = None, 
    n_trees = 500, honesty = False, mtry = 5, verbose = False, subsample_ratio = 0.5,
    oracle = False,min_leaf_size=10, max_depth=100, n_proposals = 200, 
    balancedness_tol = 0.3, bootstrap = False, seed = None):

    models = {}

    opt_solver = partial(solve_multi_nv, h_list = h_list, b_list = b_list, C = C, verbose = verbose)
    hessian_computer = partial(compute_hessian, h_list = h_list, b_list = b_list, C = C)
    gradient_computer = partial(compute_gradient, h_list = h_list, b_list = b_list, C = C)
    active_constraint = partial(search_active_constraint, C = C)
    update_step = partial(compute_update_step, constraint = True)

    forest_temp = forest(opt_solver = opt_solver, hessian_computer = hessian_computer,
                         gradient_computer = gradient_computer, 
                          search_active_constraint = active_constraint,
                         compute_update_step = update_step,
                         crit_computer = compute_crit_random, 
                         impurity_computer = partial(impurity_rf, h_list = h_list, b_list = b_list, C = C),
                        subsample_ratio=subsample_ratio, bootstrap = bootstrap, n_trees = n_trees, 
                         honesty = honesty, mtry = mtry,
                         min_leaf_size = min_leaf_size, max_depth = max_depth, 
                         n_proposals = n_proposals, balancedness_tol = balancedness_tol,
                         verbose = verbose, seed = seed)
    forest_temp.fit(Y, X, Y_est, X_est)
    models['rf_random'] = forest_temp

    forest_temp = forest(opt_solver = opt_solver, hessian_computer = hessian_computer,
                         gradient_computer = gradient_computer, 
                          search_active_constraint = active_constraint,
                         compute_update_step = update_step,
                         crit_computer = partial(compute_crit_approx_sol, h_list =  h_list, b_list = b_list),
                         impurity_computer = partial(impurity_approx_sol, h_list = h_list, b_list = b_list, C = C), 
                         subsample_ratio=subsample_ratio, bootstrap = bootstrap, n_trees = n_trees, 
                         honesty = honesty, mtry = mtry,
                         min_leaf_size = min_leaf_size, max_depth = max_depth, 
                         n_proposals = n_proposals, balancedness_tol = balancedness_tol,
                         verbose = verbose, seed = seed)
    forest_temp.fit(Y, X, Y_est, X_est)
    models['rf_approx_sol'] = forest_temp

    forest_temp = forest(opt_solver = opt_solver, hessian_computer = hessian_computer,
                         gradient_computer = gradient_computer, 
                          search_active_constraint = active_constraint,
                         compute_update_step = update_step,
                         crit_computer = compute_crit_approx_risk, 
                         impurity_computer = partial(impurity_approx_risk, h_list = h_list, b_list = b_list, C = C), 
                        subsample_ratio=subsample_ratio, bootstrap = bootstrap, n_trees = n_trees, 
                         honesty = honesty, mtry = mtry,
                         min_leaf_size = min_leaf_size, max_depth = max_depth, 
                         n_proposals = n_proposals, balancedness_tol = balancedness_tol,
                         verbose = verbose, seed = seed)
    forest_temp.fit(Y, X, Y_est, X_est)
    models['rf_approx_risk'] = forest_temp

    models['SAA'] = opt_solver(Y)[0]

    models['knn_5'] = NearestNeighbors(n_neighbors= 5, algorithm='ball_tree').fit(X)
    models['knn_10'] = NearestNeighbors(n_neighbors= 10, algorithm='ball_tree').fit(X)
    models['knn_50'] = NearestNeighbors(n_neighbors= 50, algorithm='ball_tree').fit(X)
    
    if oracle:
        forest_temp = forest(opt_solver = opt_solver, hessian_computer = hessian_computer,
                         gradient_computer = gradient_computer, 
                          search_active_constraint = active_constraint,
                         compute_update_step = update_step,
                         crit_computer = partial(compute_crit_oracle, solver = opt_solver), 
                         impurity_computer = partial(impurity_oracle, h_list = h_list, b_list = b_list, C = C), 
                        subsample_ratio=subsample_ratio, bootstrap = bootstrap, n_trees = n_trees, 
                         honesty = honesty, mtry = mtry,
                         min_leaf_size = min_leaf_size, max_depth = max_depth, 
                         n_proposals = n_proposals, balancedness_tol = balancedness_tol,
                         verbose = verbose, seed = seed)
        forest_temp.fit(Y, X, Y_est, X_est)
        models['rf_oracle'] = forest_temp

    return models

def evaluate_one_run(models, X, Y, X_est, Y_est, Nx_test, Ny_train, Ny_test, cond_mean, cond_std, 
    h_list = None, b_list = None, C = None, verbose = False, seed = None):
    
    p = X_est.shape[1]
    L = Y_est.shape[1]

    solver = partial(solve_multi_nv, h_list = h_list, b_list = b_list, C = C, verbose = verbose)
    evaluate_risk = partial(evaluate_risk_singleX, h_list = h_list, b_list = b_list)

    if seed != None:
        np.random.seed(seed)
    X_test = np.random.normal(size = (Nx_test, p))

    decisions = {}
    risk = {}

    for key in models.keys():
        decisions[key] = np.zeros((Nx_test, L))
        risk[key] = np.zeros(Nx_test)

    decisions["oracle"] = np.zeros((Nx_test, L))
    risk["oracle"] = np.zeros(Nx_test)

    for i in range(Nx_test):
        if i % 100 == 0:
            print("the", i, "th evaluation sample.")
        Y_train = generate_Y(X_test[i, :][np.newaxis, :], cond_mean, cond_std, Ny = Ny_train, seed =seed)
        Y_test = generate_Y(X_test[i, :][np.newaxis, :], cond_mean, cond_std, Ny = Ny_test, seed =seed)

        for key in models.keys():
            if (str(key).find('rf') != -1):
                try:
                    weights = models[key].get_weights(X_test[i, :]) 
                    (decisions[key][i, :], _, _, _) = solver(Y_est, weights = weights, if_weight = True)
                    risk[key][i] = evaluate_risk(decisions[key][i, :], Y_test)
                except:
                    decisions[key][i, :] = np.array([np.nan for i in range(L)])
                    risk[key][i] = np.nan
            if str(key).find('SAA') != -1: 
                (decisions[key][i, :], _) = models[key]
                risk[key][i] = evaluate_risk(decisions[key][i, :], Y_test)
            if str(key).find('knn') != -1:
                k = float(str(key).split("_")[1])
                weights = compute_knn_weights(models[key], X_test[i, :], Y.shape[0], k = k)
                (decisions[key][i, :], _, _, _) = solver(Y, weights = weights, if_weight = True)
                risk[key][i] = evaluate_risk(decisions[key][i, :], Y_test)

        decisions["oracle"][i, :] = solver(Y_train)[0]
        risk["oracle"][i] = evaluate_risk(decisions["oracle"][i, :], Y_test)

    return (decisions, risk)

def evaluate_risk_singleX(decision, Y_test, h_list, b_list):
    temp = decision[np.newaxis, :] - Y_test 
    return (np.matmul(temp * (temp >= 0), h_list) + np.matmul(-temp * (temp < 0), b_list)).mean(0)

def compute_knn_weights(nbrs, x_test, N,  k = 1):
    x_test = x_test[np.newaxis, :]
    
    w = np.zeros(N)
    (dist, ind) = nbrs.kneighbors(x_test)
    
    for i in range(x_test.shape[0]):
        w[ind[i]] = 1/k
        
    return w

def extract_risk(results_eval):
    # input: results_eval is a list of length equal to the number of runs,
    #   and each list element has two elements: 
    #       the first one being dictionary of decisions made by different models
    #       the second one being dictionary of loss of different models evaluated 
    #           at each testing example  

    exmp_key = list(results_eval[0][1])[0]
    N_test = results_eval[0][1][exmp_key].shape[0]
    runs = len(results_eval)

    loss = {key: np.zeros((N_test, runs)) for key in results_eval[0][1].keys()}

    for run in range(runs):
        (_, temp_loss) = results_eval[run]
        for key in loss.keys():
            loss[key][:, run] = temp_loss[key]

    risk = {}
    for key in loss.keys():
        risk[key] = loss[key].mean(0)
        
    return risk


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
                feature_importance[key][i, :] = results_fit[i][key].compute_impurity_fi(p)
    return feature_importance