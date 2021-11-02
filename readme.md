This repo contains code for the paper [Stochastic Optimization Forests](https://arxiv.org/abs/2008.07473). 


# Code structure 
The tree and forest classes are in tree.py, and the splitting criterion implementations for newsvendor problem, CVaR optimization, mean variance optimization, shortest path optimization are in newsvendor/nv_tree_utilities.py, cvar/cvar_tree_utilities.py, mean_var/meanvar_tree_utilities.py, and uber/cvar_tree_utilities.py, respectively. All scripts for different experiments are experiment_*.py files in each directory. Calling 'python experiment_name.py' will run these experiments in python. 

Part of the code for tree and forest classes builds on the [EconML](https://github.com/microsoft/EconML) package: 
- EconML: A Python Package for ML-Based Heterogeneous Treatment Effects Estimation. https://github.com/microsoft/EconML, 2019. Version 0.x.


# Generating the figures and tables
The basic process of generating the figures is to first run the corresponding experiment script in each directory to get experimental results stored in .pkl files, and then use prepare_plot_data.ipynb to transform the .pkl files into .csv files, and finally use the .Rmd file in each directory to generate the plots. 

## CVaR Portfolio optimization 
### Figure 2
- Figure 2(a): cvar/experiment_cvar_lognormal.py --> cvar/risk_cvar_lognormal.pkl -->  cvar/risk_lognormal.csv --> cvar/Plotting_cvar.Rmd
- Figure 2(b): cvar/experiment_cvar_lognormal.py --> cvar/feature_imp_cvar_lognormal.pkl -->
cvar/feature_imp_cvar_lognormal.csv --> cvar/Plotting_cvar.Rmd

### Figure 7 - 9
- Figure 7: cvar/experiment_cvar_lognormal.py --> cvar/feature_split_cvar_lognormal.pkl --> cvar/feature_split_cvar_lognormal.csv --> cvar/Plotting_cvar.Rmd
- Figure 8: cvar/experiment_cvar_lognormal_oracle.py --> cvar/risk_cvar_lognormal_oracle.pkl --> cvar/risk_lognormal_oracle.csv--> cvar/Plotting_cvar.Rmd
- Figire 9: cvar/experiment_cvar_lognormal_objcoef.py --> cvar/risk_cvar_lognormal_objcoef.pkl --> cvar/risk_lognormal_objcoef.csv --> cvar/Plotting_cvar.Rmd

### Figure 10 
- Figure 10(a): cvar/experiment_cvar_normal.py --> cvar/risk_cvar_normal.pkl --> cvar/risk_normal.csv--> cvar/Plotting_cvar.Rmd
- Figure 10(b): cvar/experiment_cvar_normal_oracle.py --> cvar/risk_cvar_normal_oracle.pkl --> cvar/risk_normal_oracle.csv--> cvar/Plotting_cvar.Rmd

## Uber experiment 
### Figure 3
- uber/experiment_downtown_years.py --> uber/downtown_risks_forest_years_halfyear.pkl,  uber/downtown_risks_forest_years_oneyear.pkl, uber/downtown_risks_forest_years_onehalfyear.pkl, uber/downtown_risks_forest_years_twoyear.pkl --> uber/downtown_risks_forest_years_halfyear.csv,  uber/downtown_risks_forest_years_oneyear.csv, uber/downtown_risks_forest_years_onehalfyear.csv, uber/downtown_risks_forest_years_twoyear.csv --> 
Plotting_uber.Rmd
- All raw data files are in uber/data. 
- See uber/data_downloading.R and uber/preprocessing.R for data collection and preprocessing.  

## Newsvendor 
### Figure 5
- Fig 5(a) newsvendor/experiment_nv_n.py --> newsvendor/risk_n.pkl -->  newsvendor/risk_nv_n.csv --> newsvendor/Plotting_newsvendor.Rmd
- Fig 5(b) newsvendor/experiment_nv_n.py --> newsvendor/feature_split_n.pkl -->  newsvendor/feature_split_nv_n.csv --> newsvendor/Plotting_newsvendor.Rmd
- Fig 5(c) newsvendor/experiment_nv_n.py --> newsvendor/feature_importance_n.pkl -->  newsvendor/feature_importance_n.csv --> newsvendor/Plotting_newsvendor.Rmd

### Figure 6
- Fig 6(a) newsvendor/experiment_nv_p.py --> newsvendor/risk_p.pkl --> newsvendor/risk_nv_p.csv --> newsvendor/Plotting_newsvendor.Rmd
- Fig 6(b) newsvendor/experiment_nv_highdim.py --> newsvendor/risk_highdim.pkl -->  newsvendor/risk_highdim.csv --> newsvendor/Plotting_newsvendor.Rmd

## mean-variance optimization
### Figure 4
- Fig 4(a): mean_var/experiment_meanvar_stoch.py --> mean_var/rel_risk_meanvar_normal_stoch.pkl -->  mean_var/rel_risk_full.csv --> mean_var/Plotting_meanvar.Rmd
- Fig 4(b): mean_var/experiment_meanvar_stoch.py --> mean_var/feature_split_meanvar_normal_stoch.pkl --> mean_var/feature_freq_full.csv --> mean_var/Plotting_meanvar.Rmd
- Fig 4(c): mean_var/experiment_meanvar_stoch.py --> mean_var/cond_violation_meanvar_normal_stoch.pkl --> mean_var/cond_violation_full.csv --> mean_var/Plotting_meanvar.Rmd
- Fig 4(d): mean_var/experiment_meanvar_stoch.py --> mean_var/mean_violation_meanvar_normal_stoch.pkl --> mean_var/marginal_violation_full.csv --> mean_var/Plotting_meanvar.Rmd

### Figure 12
- Fig 12(a): mean_var/experiment_var_normal_oracle.py --> mean_var/risk_var_normal_oracle.pkl --> mean_var/risk_var_normal_oracle.csv --> mean_var/Plotting_var.Rmd
- Fig 12(b): mean_var/experiment_var_normal_oracle.py --> mean_var/feature_split_var_normal_oracle.pkl --> mean_var/feature_split_var_normal_oracle.csv --> mean_var/Plotting_var.Rmd
- Fig 12(c): mean_var/experiment_var_normal.py --> mean_var/risk_var_normal.pkl --> mean_var/risk_var_normal.csv --> mean_var/Plotting_var.Rmd

### Figure 13
- Fig 13(a): mean_var/experiment_meanvar_stoch_oracle.py --> mean_var/rel_risk_meanvar_normal_stoch_oracle.pkl -->  mean_var/rel_risk_full_oracle.csv --> mean_var/Plotting_meanvar.Rmd
- Fig 13(b): mean_var/experiment_meanvar_stoch.py --> mean_var/rel_risk_meanvar_normal_stoch_oracle.pkl --> mean_var/feature_freq_full_oracle.csv --> mean_var/Plotting_meanvar.Rmd

### Figure 14
- Fig 14: mean_var/experiment_meanvar_stoch_R.py --> mean_var/rel_risk_meanvar_normal_stoch_R.pkl -->  mean_var/rel_risk_full_R.csv --> mean_var/Plotting_meanvar.Rmd

## honest forests
- Fig 15(a): cvar/experiment_cvar_lognormal_honesty.py --> cvar/risk_cvar_lognormal_honesty.pkl --> cvar/risk_lognormal_honesty.csv --> cvar/Plotting_cvar.Rmd
- Fig 15(b): newsvendor/experiment_nv_honesty.py --> newsvendor/risk_nv_honesty.pkl --> newsvendor/risk_nv_honesty.csv --> newsvendor/Plotting_newsvendor.Rmd

## Running time 
- Table 1: cvar/speed_cvar.ipynb --> time_cvar.pkl
- Table 2: mean_var/speed_meanvar.ipynb --> time_meanvar.pkl

# Dependencies
## python 3.6.10
- gurobipy                  9.0.2
- joblib                    0.16.0
- numpy                     1.19.1
- scikit-learn              0.23.2
- scipy                     1.3.1
## R 3.6.1
- latex2exp 0.4.0
- tidyverse 1.3.0

