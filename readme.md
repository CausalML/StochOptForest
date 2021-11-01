This repo contains code for the paper [Stochastic Optimization Forests](https://arxiv.org/abs/2008.07473). 


# Code structure 
The tree and forest classes are in tree.py, and the splitting criterion implementations for newsvendor problem, CVaR optimization and mean variance optimization are in newsvendor/nv_tree_utilities.py, cvar/cvar_tree_utilities.py and mean_var/meanvar_tree_utilities.py respectively. All scripts for different experiments are experiment_*.py files in each directory. Use 'python experiment_name.py' will run these experiments in python. 

Part of the code for tree and forest classes builds on the [EconML](https://github.com/microsoft/EconML) package: 
- EconML: A Python Package for ML-Based Heterogeneous Treatment Effects Estimation. https://github.com/microsoft/EconML, 2019. Version 0.x.


# Generating the figures and tables
The basic process of generating the figures is to first run the corresponding experiment script in each directory to get experimental results stored in .pkl files, and then use prepare_plot_data.ipynb to transform the .pkl files into .csv files, and finally use the .Rmd file in each directory to generate the plots. 

## newsvendor problem 
### Figure 2
- Fig 2(a) newsvendor/experiment_nv_n.py --> newsvendor/risk_n.pkl -->  newsvendor/risk_nv_n.csv --> newsvendor/Plotting_newsvendor.Rmd
- Fig 2(b) newsvendor/experiment_nv_n.py --> newsvendor/feature_split_n.pkl -->  newsvendor/feature_split_nv_n.csv --> newsvendor/Plotting_newsvendor.Rmd
- Fig 2(c) newsvendor/experiment_nv_p.py --> newsvendor/risk_p.pkl -->  newsvendor/risk_nv_p.csv --> newsvendor/Plotting_newsvendor.Rmd

## CVaR optimization 
- Figure 3: cvar/experiment_cvar_lognormal.py --> cvar/risk_cvar_lognormal.pkl -->  cvar/risk_lognormal.csv --> cvar/Plotting_cvar.Rmd
- Figure 5: cvar/experiment_cvar_lognormal_oracle.py --> cvar/risk_cvar_lognormal_oracle.pkl --> cvar/risk_lognormal_oracle.csv--> cvar/Plotting_cvar.Rmd
- Figire 6: cvar/experiment_cvar_lognormal_objcoef.py --> cvar/risk_cvar_lognormal_objcoef.pkl --> cvar/risk_lognormal_objcoef.csv --> cvar/Plotting_cvar.Rmd

### Figure 7 
- Figure 7(a): cvar/experiment_cvar_normal.py --> cvar/risk_cvar_normal.pkl --> cvar/risk_normal.csv--> cvar/Plotting_cvar.Rmd
- Figure 7(b): cvar/experiment_cvar_normal_oracle.py --> cvar/risk_cvar_normal_oracle.pkl --> cvar/risk_normal_oracle.csv--> cvar/Plotting_cvar.Rmd


## mean-variance optimization
### Figure 4
- Fig 4(a): mean_var/experiment_meanvar_stoch.py --> mean_var/rel_risk_meanvar_normal_stoch.pkl -->  mean_var/rel_risk_full.csv --> mean_var/Plotting_meanvar.Rmd
- Fig 4(b): mean_var/experiment_meanvar_stoch.py --> mean_var/feature_split_meanvar_normal_stoch.pkl --> mean_var/feature_freq_full.csv --> mean_var/Plotting_meanvar.Rmd
- Fig 4(c): mean_var/experiment_meanvar_stoch.py --> mean_var/cond_violation_meanvar_normal_stoch.pkl --> mean_var/cond_violation_full.csv --> mean_var/Plotting_meanvar.Rmd
- Fig 4(d): mean_var/experiment_meanvar_stoch.py --> mean_var/mean_violation_meanvar_normal_stoch.pkl --> mean_var/marginal_violation_full.csv --> mean_var/Plotting_meanvar.Rmd

### Figure 8
- Fig 8(a): mean_var/experiment_var_normal_oracle.py --> mean_var/risk_var_normal_oracle.pkl --> mean_var/risk_var_normal_oracle.csv --> mean_var/Plotting_var.Rmd
- Fig 8(b): mean_var/experiment_var_normal_oracle.py --> mean_var/feature_split_var_normal_oracle.pkl --> mean_var/feature_split_var_normal_oracle.csv --> mean_var/Plotting_var.Rmd
- Fig 8(c): mean_var/experiment_var_normal.py --> mean_var/risk_var_normal.pkl --> mean_var/risk_var_normal.csv --> mean_var/Plotting_var.Rmd

### Figure 9 
- Fig 9(a): mean_var/experiment_meanvar_stoch_oracle.py --> mean_var/rel_risk_meanvar_normal_stoch_oracle.pkl -->  mean_var/rel_risk_full_oracle.csv --> mean_var/Plotting_meanvar.Rmd
- Fig 9(b): mean_var/experiment_meanvar_stoch.py --> mean_var/rel_risk_meanvar_normal_stoch_oracle.pkl --> mean_var/feature_freq_full_oracle.csv --> mean_var/Plotting_meanvar.Rmd

### Figure 10
- Fig 10(a): mean_var/experiment_meanvar_stoch_R.py --> mean_var/rel_risk_meanvar_normal_stoch_R.pkl -->  mean_var/rel_risk_full_R.csv --> mean_var/Plotting_meanvar.Rmd
- Fig 10(b): mean_var/experiment_meanvar_stoch.py --> mean_var/risk_meanvar_normal_stoch.pkl --> mean_var/abs_risk_full.csv --> mean_var/Plotting_meanvar.Rmd

## honest forests
- Fig 11(a): cvar/experiment_cvar_lognormal_honesty.py --> cvar/risk_cvar_lognormal_honesty.pkl --> cvar/risk_lognormal_honesty.csv --> cvar/Plotting_cvar.Rmd
- Fig 11(b): newsvendor/experiment_nv_honesty.py --> newsvendor/risk_nv_honesty.pkl --> newsvendor/risk_nv_honesty.csv --> newsvendor/Plotting_newsvendor.Rmd

## Running time 
- Table 1: cvar/speed_cvar.ipynb --> time_cvar.pkl
- Table 2: mean_var/speed_meanvar.ipynb --> time_meanvar.pkl


# Dependencies
## python 3.6.10
- gurobipy                  8.1.1
- joblib                    0.16.0
- numpy                     1.19.1
- scikit-learn              0.23.2
- scipy                     1.3.1
## R 3.6.1
- latex2exp 0.4.0
- tidyverse 1.3.0

