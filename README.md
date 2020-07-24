# serialNMDA
This repository contains all codes necessary to reproduce figures and results reported in Stein, Barbosa et al. (Nature Communications, in press, 2020) from the raw data acquired in human behavioral experiments (data included in the repository), and from the relevant model simulations.

## data
data/ contains 
-   behavioral data from the baseline session for all subjects ("behavior.pkl"), and behavioral data for all subjects that completed baseline and follow-up sessions ("behavior_retest.pkl")
-   "DOG1_par_exp_1000.npy" and "DOG1_par_sims_1000.npy" store the optimal parameter values found by crossvalidation ("codes/model_fits_behavior.py" and "codes/model_fits_modeling.py") for the derivative of Gaussian (DoG) basis function used to model serial dependence in the experimental data and the simulations, respectively. 
-   subfolders "Fig_x/" contains preprocessed data for plotting, used by scripts "codes/FIGUREx.py"

*please note: raw simulation outputs for all parameter values (as used as inputs to "model_fits_modeling.py") imply large files that are not included in this repository. All simulations can be reproduced with codes or sent upon request*

## codes
codes/ contains 
-   codes that plot main figures 1-3: "FIGUREx.py"
-   frequently used functions: "helpers.py"
-   code that performs crossvalidation of hyperparameter sigma in DoG basisfunction for experimental data: "model_fits_behavior.py"
-   code that performs crossvalidation of hyperparameter sigma in DoG basisfunction for simulations, and estimates bias strength across all simulated parameter values: "model_fits_modeling.py"
-   code that estimates all models of behavior for the baseline session, reported in the main text: "models_behavior.py"
-   code that estimates longitudinal effects in behavior: "models_behavior_retest.py"
-   codes used to run simulations on cluster: "wm_model_input_Apre.py", "wm_model_input_gei.py", "wm_model_input_gee.py"
-   code that runs a single simulation including figures: "wm_model_stp_modeling.py"

*please note: raw simulation outputs for all parameter values (as used as inputs to "model_fits_modeling.py") imply large files that are not included in this repository. All simulations can be reproduced with codes or sent upon request*

## questions?
If you still have open questions, do not hesitate to contact [Albert Compte](albert.compte@clinic.cat) (corresponding author) or [Heike Stein](heike.c.stein@gmail.com) (for codes and data)
