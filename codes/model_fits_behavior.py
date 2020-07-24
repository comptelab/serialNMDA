# MODEL SELECTION AND HYPERPAR. CROSSVALIDATION, BEHAVIOR

# author: heike stein
# last mod: 23/04/20

import numpy as np
import pandas as pd
import json
from joblib import Parallel, delayed
import multiprocessing
import statsmodels.api as sm
from patsy import dmatrices
import helpers as hf

numcores    = multiprocessing.cpu_count() # for parallel processing


##############################################################################
#                       LOAD AND PREPARE DATASET                             #
##############################################################################

# load first session data
alldat      = pd.read_pickle('../data/behavior.pkl')

# exclude outliers and first trials of each block (blocklength = 48 trials)
dat         = hf.filter_dat(alldat, rt=3, iti=5, raderr=5, err=1)
first       = dat.trial%48==1   
dat         = dat[~first]       # delete first trial of each block


##############################################################################
#               EXPLORE GROUP MODELS FOR HYPERPARAMETERS                     #
##############################################################################

# dog(prevcurr) (3rd derivative of Gaussian) model with different sigma hyperparameters
aic_dog3 = []; sse_dog3 = []; s_dog3 = []
for s in np.arange(.6,2.0,.05):
    print 'crossvalidating 3rd derivative of Gaussian fits', s
    # fit model and save AIC
    y, X    = dmatrices('error ~ group*hf.dog3(s,prevcurr)*delay', 
              data=dat, return_type='dataframe')
    glm     = sm.OLS(y, X).fit()
    aic_dog3.append(glm.aic)
    # cross-validate and save SSE
    sse     = Parallel(n_jobs=numcores)(delayed(hf.cross_validate)(y,X,dat.subject) for i in range(1000))
    sse_dog3.append(np.mean(sse))
    s_dog3.append(s)

# dog(prevcurr) (1st derivative of Gaussian) model with different sigma hyperparameters
aic_dog1 = []; sse_dog1 = []; s_dog1 = []
for s in np.arange(.2,1.8,.05):
    print 'crossvalidating 1st derivative of Gaussian fits', s
    # fit model and save AIC
    y, X    = dmatrices('error ~ group*hf.dog1(s,prevcurr)*delay', 
              data=dat, return_type = 'dataframe')
    glm     = sm.OLS(y, X).fit()
    aic_dog1.append(glm.aic)
    # cross-validate and save SSE
    sse     = Parallel(n_jobs=numcores)(delayed(hf.cross_validate)(y,X,dat.subject) for i in range(1000))
    sse_dog1.append(np.mean(sse))
    s_dog1.append(s)

# # find out best model/hyperparameter from crossvalidated SSE and add line to dataframe
if sse_dog1[np.argmin(sse_dog1)] < sse_dog3[np.argmin(sse_dog3)]:
    print '1st derivative of Gaussian fits data best'
else:
    print '3rd derivative of Gaussian fits data best'

sses = {'dog1': dict(zip(np.around(s_dog1,3), sse_dog1)), 'dog3': dict(zip(np.around(s_dog3,3), sse_dog3))}

# # save stuff for supplementary figure 1
# np.save('../data/Supplementary_Fig_1/SUPPLEMENTARY_FIGURE1_dog_pars_exp1_1000.npy', [sses])
# np.save('../data/DoG1_par_exp1_1000.npy',np.around(s_dog1[np.argmin(sse_dog1)],3))
# np.save('../data/DoG3_par_exp1_1000.npy',np.around(s_dog3[np.argmin(sse_dog3)],3))
