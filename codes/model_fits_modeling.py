# HYPERPAR. CROSSVALIDATION AND LINEAR MODEL FITS, SIMULATIONS

# author: heike stein
# last mod: 23/04/20

import numpy as np
from glob import glob
import scikits.bootstrap as boot
import scipy.stats as sps
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrices
import helpers as hf
from joblib import Parallel, delayed
import multiprocessing

numcores    = multiprocessing.cpu_count() 

#########################################################################
#                          CROSSVALIDATION                              #
#########################################################################

names = '../simulations/STP_EE/*beh*Apre0.00022*1.25.txt'

shortdat = pd.DataFrame(np.loadtxt(glob(names)[0], dtype='S')[:20000,1:].astype('float'), \
           columns=['previous','current','resp0','resp1','resp3'])
shortdat['trial'] = range(1, len(shortdat)+1)

dat = pd.wide_to_long(shortdat, stubnames=['resp'], i='trial', 
      j='delay', sep='').reset_index(level=['trial', 'delay'])
dat['prevcurr'] = hf.circdist(dat.previous.values, dat.current.values)
dat['error'] = hf.circdist(dat.resp.values, dat.current.values)
dat = dat[abs(dat.error)<1]

# crossvalidate sigma hyperparameters
aic_fit = []; sse_fit = []; par = []
for s in np.arange(.2,1.8,.05):
    print('crossvalidating 1st derivative of Gaussian fits', s)
    # fit model and save AIC
    y, X    = dmatrices('error ~ hf.dog1(s,prevcurr)*delay', 
              data=dat, return_type='dataframe')
    glm     = sm.OLS(y, X).fit()
    aic_fit.append(glm.aic)
    # cross-validate and save SSE
    sse     = Parallel(n_jobs=numcores)(delayed(hf.cross_validate)(y,X) for i in range(1000))
    sse_fit.append(np.mean(sse))
    par.append(s)

par = par[np.argmin(sse_fit)]

#np.save('../data/DOG1_par_sims_1000.npy', [sse_fit])


#########################################################################
#                            BIAS ESTIMATES                             #
#########################################################################

par = np.load('../data/DOG1_par_sims_1000.npy')
par = np.arange(.2,1.8,.05)[np.argmin(par)]

for name in ['Apre','gee','gei']:
    behs = sorted(glob('../simulations/STP_EE/read_out_log_beh*' + name + '*1.25*'))
    # behs = sorted(glob('../simulations/STP_in_EE_EI/read_out_log_beh*' + name + '*1.25*'))
    # print(behs)
    bias=[]; bias_ci=[]; prec=[]; prec_ci=[]; outliers=[]
    for beh in behs:
        print(beh)
        sims    = pd.DataFrame(np.loadtxt(beh, dtype='S')[:,1:].astype('float'),
                  columns=['previous','current','resp0','resp1','resp3'])[:20000]
        sims['trial'] = range(1, len(sims)+1)
        dat     = pd.wide_to_long(sims, stubnames=['resp'], i='trial', 
                  j='delay', sep='').reset_index(level=['trial', 'delay'])
        dat['prevcurr'] = hf.circdist(dat.previous.values, dat.current.values)
        dat['error'] = hf.circdist(dat.resp.values, dat.current.values)

        outliers.append(sum(((abs(dat.error)>1) & (dat.delay==3)))/float(sum(dat.delay==3))*100)
        dat     = dat[abs(dat.error)<1]

        y, X    = dmatrices('error ~ hf.dog1(par,prevcurr)', data=dat[dat.delay==0], 
                  return_type='dataframe')
        glm0    = sm.OLS(y, X).fit()
        y, X    = dmatrices('error ~ hf.dog1(par,prevcurr)', data=dat[dat.delay==1], 
                  return_type='dataframe')
        glm1    = sm.OLS(y, X).fit()
        y, X    = dmatrices('error ~ hf.dog1(par,prevcurr)', data=dat[dat.delay==3], 
                  return_type='dataframe')
        glm3    = sm.OLS(y, X).fit()
        
        bias.append(-np.rad2deg([glm0.params[1],glm1.params[1],glm3.params[1]]))
        bias_ci.append(-np.rad2deg(np.array([glm0.conf_int().iloc[1,:],
                  glm1.conf_int().iloc[1,:], glm3.conf_int().iloc[1,:]])))
        
        prec.append(np.rad2deg(sps.circstd(glm3.resid)))
        prec_ci.append(np.rad2deg(boot.ci(glm3.resid, statfunction=sps.circstd)))


    # np.save('../data/Fig_3/SUPPLEMENTARY_FIGURE14_bias_' + name + '.npy', bias)
    # np.save('../data/Fig_3/SUPPLEMENTARY_FIGURE14_bias_ci_' + name + '.npy', bias_ci)
    # np.save('../data/Fig_3/SUPPLEMENTARY_FIGURE14_prec_' + name + '.npy', np.column_stack((np.array(prec), np.array(prec_ci))))
    # np.save('../data/Fig_3/SUPPLEMENTARY_FIGURE14_outliers_' + name + '.npy', outliers)
    # if name == 'Apre':
    #   np.save('../data/Fig_3/SUPPLEMENTARY_FIGURE14_xticks_Apre.npy', [beh[-14:-12] for beh in behs][::4])
    # else:
    #   np.save('../data/Fig_3/SUPPLEMENTARY_FIGURE14_xticks_' + name + '.npy', [np.around((float(beh[-19:-14])-1)*100,3) for beh in behs][::4])