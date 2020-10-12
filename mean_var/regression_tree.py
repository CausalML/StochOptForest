import numpy as np
import warnings
from scipy.stats import gaussian_kde
import time
from joblib import Parallel, delayed
from functools import partial
from functools import reduce
from statsmodels.nonparametric.kernel_regression import KernelReg
from sklearn.neighbors import NearestNeighbors
import pickle 
import scipy.stats 
from scipy.stats import truncnorm
import gurobipy as gp
from gurobipy import GRB
import pandas as pd


def compute_crit_rf(node_Y, valid_side):

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

def generate_candidate_splits(node_X, node_X_estimate, mtry, n_proposals, balancedness_tol, min_leaf_size):

    dim_proposals = np.random.choice(node_X.shape[1], mtry, replace = False)
    dim_proposals = np.array([dim for dim in dim_proposals for i in range(n_proposals)])
    thr_inds = np.random.choice(node_X.shape[0], n_proposals, replace = False)
    thr_inds = np.array(list(thr_inds) * mtry)
    thr_proposals = node_X[thr_inds, dim_proposals]


    # calculate the binary indicator of whether sample i is on the left or the right
    # side of proposed split j. So this is an n_samples x n_proposals matrix
    side = node_X[:, dim_proposals] < thr_proposals
    # calculate the number of samples on the left child for each proposed split
    size_left = np.sum(side, axis=0)
    # calculate the analogous binary indicator for the samples in the estimation set
    side_est = node_X_estimate[:, dim_proposals] < thr_proposals
    # calculate the number of estimation samples on the left child of each proposed split
    size_est_left = np.sum(side_est, axis=0)



    # find the upper and lower bound on the size of the left split for the split
    # to be valid so as for the split to be balanced and leave at least min_leaf_size
    # on each side.
    lower_bound = max((balancedness_tol) * node_X.shape[0], min_leaf_size)
    upper_bound = min((1 - balancedness_tol) * node_X.shape[0], node_X.shape[0] - min_leaf_size)
    valid_split = (lower_bound <= size_left)
    valid_split &= (size_left <= upper_bound)

    # similarly for the estimation sample set
    lower_bound_est = max((balancedness_tol) * node_X_estimate.shape[0], min_leaf_size)
    upper_bound_est = min((1 -  balancedness_tol) * node_X_estimate.shape[0], node_X_estimate.shape[0] - min_leaf_size)
    valid_split &= (lower_bound_est <= size_est_left)
    valid_split &= (size_est_left <= upper_bound_est)

    if ~np.any(valid_split):
        return (None, None, None, None)

    # filter only the valid splits
    valid_dim_proposals = dim_proposals[valid_split]
    valid_thr_proposals = thr_proposals[valid_split]
    valid_side = side[:, valid_split]
    valid_size_left = size_left[valid_split]
    valid_side_est = side_est[:, valid_split]

    return (valid_side, valid_side_est, valid_dim_proposals, valid_thr_proposals)

class Node:
    """Building block of :class:`CausalTree` class.

    Parameters
    ----------
    sample_inds : array-like, shape (n, )
        Indices defining the sample that the split criterion will be computed on.

    estimate_inds : array-like, shape (n, )
        Indices defining the sample used for calculating balance criteria.

    """

    def __init__(self, sample_inds, estimate_inds):
        self.feature = -1
        self.threshold = np.inf
        self.split_sample_inds = sample_inds
        self.est_sample_inds = estimate_inds
        self.left = None
        self.right = None
        self.hessian = None 

    def find_tree_node(self, value):
        """
        Recursively find and return the node of the causal tree that corresponds
        to the input feature vector.

        Parameters
        ----------
        value : array-like, shape (d_x,)
            Feature vector whose node we want to find.
        """
        if self.feature == -1:
            return self
        elif value[self.feature] < self.threshold:
            return self.left.find_tree_node(value)
        else:
            return self.right.find_tree_node(value)

def build_reg_tree(Y, X, Y_est, X_est, 
                         honesty = False, mtry = 10,
                         min_leaf_size = 10, max_depth = 100, 
                         n_proposals = 200, balancedness_tol = 0.3,
                         verbose = False):
    tree_temp =  reg_tree(honesty = honesty, mtry = mtry,
                         min_leaf_size = min_leaf_size, max_depth = max_depth, 
                         n_proposals = n_proposals, balancedness_tol = balancedness_tol,
                         verbose = verbose)

    tree_temp.create_splits(Y, X, Y_est, X_est)
    return tree_temp

class reg_tree:
    def __init__(self,
                 honesty = False,
                 min_leaf_size=10,
                 max_depth=10,
                 n_proposals=200,
                 mtry = 5,
                 balancedness_tol=.3, 
                 verbose = False
                 ):
        # tree parameters
        self.min_leaf_size = min_leaf_size
        self.max_depth = max_depth
        self.balancedness_tol = balancedness_tol
        self.n_proposals = n_proposals
        self.honesty = honesty
        self.min_leaf_size = min_leaf_size
        self.max_depth = max_depth
        self.mtry = mtry
        # Tree structure
        self.tree = None
        self.verbose = verbose 

    def create_splits(self, Y, X, Y_est, X_est):
        if self.verbose:
            print("start splitting!")
            print("------------")
        
        n = Y.shape[0]
        n_est = Y_est.shape[0]
        self.tree = Node(np.arange(n), np.arange(n_est))
        node_list = [(self.tree, 0)]
        
        while len(node_list) > 0:
            node, depth = node_list.pop()

            if self.verbose:
                print("depth:", depth)
            
            # If by splitting we have too small leaves or if we reached the maximum number of splits we stop
            if node.split_sample_inds.shape[0] >= self.min_leaf_size and depth < self.max_depth:

                # Create local sample set
                node_X = X[node.split_sample_inds]
                node_Y = Y[node.split_sample_inds]
                node_X_estimate = X_est[node.est_sample_inds]
                
                if self.verbose:
                    print("node sample size: ", node_Y.shape[0])

                # (node_sol, nu0, lambda0, _) = self.opt_solver(node_Y, verbose = self.verbose)
                

                # if node_sol is None:
                #     if self.verbose:
                #         print("node 0 optimization error!")
                #     node.opt_error = True
                #     continue 

                # node_hessian = self.hessian_computer(node_Y, node_sol)
                # (node_obj_gradient, node_constr_gradient_de, node_constr_gradient_st)= self.gradient_computer(node_Y, node_sol)
                # (active_const_de, active_const_st) = self.search_active_constraint(node_Y, node_sol, verbose = self.verbose)
                                
                # node.active_const_de = active_const_de
                # node.active_const_st = active_const_st
                # node.nu0 = nu0
                # node.lambda0 = lambda0

                n_proposals = min(self.n_proposals, node_X.shape[0])

                (valid_side, valid_side_est, valid_dim_proposals, valid_thr_proposals) = generate_candidate_splits(node_X, node_X_estimate, self.mtry, n_proposals, self.balancedness_tol, self.min_leaf_size)
                if valid_side is None:
                    continue

                if self.verbose:
                    print("Hessian:", node_hessian)


                # (node_h_left, node_h_right, f_grad, g_grad) = self.compute_update_step(node_Y, node_sol, nu0, lambda0, 
                        # node_hessian, node_obj_gradient, node_constr_gradient_de, node_constr_gradient_st,
                        # active_const_de, active_const_st, valid_side)
                (split_scores, _, _) = compute_crit_rf(node_Y, valid_side)
                if split_scores is None:
                    if self.verbose:
                        print("criterion computing error!")
                    continue 

                # best split proposal 
                best_split_ind = np.argmin(split_scores)
                node.feature = valid_dim_proposals[best_split_ind]
                node.threshold = valid_thr_proposals[best_split_ind]
                if self.verbose:
                    print("feature split: ", node.feature)
                    print("threshold split: ", node.threshold)
                
                # Create child nodes with corresponding subsamples
                left_split_sample_inds = node.split_sample_inds[valid_side[:, best_split_ind]]
                left_est_sample_inds = node.est_sample_inds[valid_side_est[:, best_split_ind]]
                node.left = Node(left_split_sample_inds, left_est_sample_inds)
                right_split_sample_inds = node.split_sample_inds[~valid_side[:, best_split_ind]]
                right_est_sample_inds = node.est_sample_inds[~valid_side_est[:, best_split_ind]]
                node.right = Node(right_split_sample_inds, right_est_sample_inds)
                
                # add the created children to the list of not yet split nodes
                node_list.append((node.left, depth + 1))
                node_list.append((node.right, depth + 1))

            self.depth = depth
                
    def print_tree_rec(self, node):
        if not node:
            return
        print("Node: ({}, {})".format(node.feature, node.threshold))
        print("Left Child")
        self.print_tree_rec(node.left)
        print("Right Child")
        self.print_tree_rec(node.right)

    def print_tree(self):
        self.print_tree_rec(self.tree)

    def find_split(self, value):
        return self.tree.find_tree_node(value.astype(np.float64))

    def find_feature(self, node):
        if node.feature is -1:
            return []

        feature_list = [node.feature] + self.find_feature(node.left) + self.find_feature(node.right)
        
        return feature_list

    def find_opt_error(self, node):
        if node.feature is -1:
            return [node.opt_error]

        opt_error_list = [node.opt_error] + self.find_opt_error(node.left) + self.find_opt_error(node.right)

        return opt_error_list

    def find_splitting_feature_depth(self, node, depth = None):
        if node.feature is -1:
            return []
        if (depth == 0) and node.feature != -1:
            return [node.feature]

        feature_list = self.find_splitting_feature_depth(node.left, depth - 1) + self.find_splitting_feature_depth(node.right, depth - 1)
        
        return feature_list

    def find_feature_depth(self, node, depth = None):
        if node.feature is -1:
            return [-1]
        if (depth == 0) and node.feature != -1:
            return [node.feature]

        feature_list = self.find_feature_depth(node.left, depth - 1) + self.find_feature_depth(node.right, depth - 1)
        
        return feature_list

    def find_threshold_depth(self, node, depth = None):
        if node.feature is -1:
            return [node.threshold]
        if (depth == 0) and node.feature != -1:
            return [node.threshold]
        threshold_list = self.find_threshold_depth(node.left, depth - 1) + self.find_threshold_depth(node.right, depth - 1)
        return threshold_list

    def find_nodesize_depth(self, node, depth = None):
        if node is None:
            return []
        if (depth == 0):
            return [len(node.split_sample_inds)]
        if (node.feature == -1):
            return [len(node.split_sample_inds)]
        
        nodesize_list = self.find_nodesize_depth(node.left, depth - 1) + self.find_nodesize_depth(node.right, depth - 1)
        return nodesize_list

    def feature_frequency_depth(self, node, p, depth = None):
        feature_list = self.find_feature_depth(node, depth = depth)

        freq = np.zeros(p + 1)
        for i in range(len(feature_list)):
            freq[feature_list[i] + 1] = freq[feature_list[i] + 1] + 1/len(feature_list)
        freq = pd.Series(freq)
        freq.index = range(-1, len(freq)-1)

        return freq

    def feature_frequency_depth_specific(self, node, p, feature = None, depth = None):
        freq = self.feature_frequency_depth(node, p, depth = depth)
        return freq[feature]


