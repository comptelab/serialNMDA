import numpy as np
import scipy.stats as sps
import pandas as pd
import helpers as hf

# use R packages in python code. You have to have R installed, and the specific
# packages also need to be installed in R
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

pandas2ri.activate()

base     = importr('base')
car      = importr('car')
stats    = importr('stats')
lme4     = importr('lme4')
scales   = importr('scales')
lmerTest = importr('lmerTest')
optimx   = importr('optimx')
mumin    = importr('MuMIn')

# specify optimizer for mixed models
ro.r("optctrl=lmerControl(optimizer='optimx', check.conv.grad=.makeCC('warning', tol=3e-3, relTol=NULL), optCtrl=list(method='nlminb'))")

##############################################################################
#                       LOAD AND PREPARE DATASET                             #
##############################################################################

# load first session data with neuropsychology attached
alldat      = pd.read_pickle('../data/behavior.pkl')
par         = np.load('../data/'+'DoG1_par_exp1_1000.npy')

# exclude outliers and first trials of each block (blocklength = 48 trials)
dat         = hf.filter_dat(alldat, rt=3, iti=5, raderr=5, err=1)
first       = dat.trial%48==1   
dat         = dat[~first].reset_index(drop=True)  # delete first trial of each block

# add DoG(prevcurr) to dataframe
# hyperparameter and DoG1 vs DoG3 from model_selection.py
dat['baseprevcurr_1']  = hf.dog1(par, dat.prevcurr_1.values)
dat['baseprevcurr']  = hf.dog1(par, dat.prevcurr.values)
dat['baseprevcurr2']  = hf.dog1(par, dat.prevcurr2.values)
 
 # create R dataframe for mixed models
ro.globalenv['rdat'] = ro.conversion.py2rpy(dat)


##############################################################################
#                       GROUP MODELS: MIXED EFFECTS                          #
##############################################################################

# decide on random effects structure
m0 = ro.r('m0 = lmer(error ~ group*baseprevcurr*delay + (1|subject), data=rdat, control=optctrl)')
m1 = ro.r('m1 = lmer(error ~ group*baseprevcurr*delay + (baseprevcurr:delay|subject), data=rdat, control=optctrl)')
ro.r("anova(m1,m0)")
#       Df           AIC           BIC        logLik      deviance       Chisq  Chi Df    Pr(>Chisq)
# m0  20.0 -78251.251346 -78073.920399  39145.625673 -78291.251346         NaN     NaN           NaN
# m1  29.0 -78673.100940 -78415.971067  39365.550470 -78731.100940  439.849594     9.0  4.237646e-89


lmer = ro.r('rlmer = lmer(error ~ group*baseprevcurr*delay + (baseprevcurr:delay|subject), data=rdat, control=optctrl)')
coefs = ro.r('data.frame(coef(summary(rlmer)))')
print(lmerTest.summary_lmerModLmerTest(lmer))
#                                Estimate Std. Error         df t value Pr(>|t|)
# (Intercept)                   5.409e-03  3.181e-03  1.068e+02   1.701  0.09190
# groupE                        4.294e-03  4.718e-03  1.080e+02   0.910  0.36471
# groupS                        1.389e-03  4.685e-03  1.121e+02   0.296  0.76745
# baseprevcurr                  8.096e-03  4.918e-03  6.170e+01   1.646  0.10483
# delay180                      1.321e-02  2.793e-03  5.228e+04   4.728 2.27e-06
# delay60                       3.844e-03  2.212e-03  5.224e+04   1.738  0.08228
# groupE:baseprevcurr           7.770e-04  7.343e-03  6.391e+01   0.106  0.91606
# groupS:baseprevcurr           8.056e-03  7.288e-03  6.610e+01   1.105  0.27299
# groupE:delay180              -1.201e-03  4.171e-03  5.224e+04  -0.288  0.77347
# groupS:delay180              -6.762e-03  4.197e-03  5.228e+04  -1.611  0.10715
# groupE:delay60                7.063e-03  3.299e-03  5.223e+04   2.141  0.03226
# groupS:delay60                1.915e-03  3.319e-03  5.223e+04   0.577  0.56402
# baseprevcurr:delay180        -4.766e-02  6.389e-03  5.462e+01  -7.459 6.94e-10
# baseprevcurr:delay60         -1.140e-02  4.552e-03  8.361e+01  -2.504  0.01424
# groupE:baseprevcurr:delay180  2.893e-02  9.558e-03  5.654e+01   3.027  0.00372
# groupS:baseprevcurr:delay180  5.433e-02  9.515e-03  5.884e+01   5.710 3.93e-07
# groupE:baseprevcurr:delay60   5.884e-03  6.815e-03  8.736e+01   0.863  0.39028
# groupS:baseprevcurr:delay60   2.051e-02  6.798e-03  9.174e+01   3.018  0.00330
print(lmerTest.anova_lmerModLmerTest(lmer))
#                             Sum Sq   Mean Sq  NumDF         DenDF    F value        Pr(>F)
# group                     0.042382  0.021191      2     54.380939   1.634800  2.044280e-01
# baseprevcurr              0.016487  0.016487      1     49.284746   1.271939  2.648638e-01
# delay                     0.505039  0.252519      2  52264.768176  19.480937  3.488901e-09
# group:baseprevcurr        0.250963  0.125482      2     49.265813   9.680449  2.845469e-04
# group:delay               0.162640  0.040660      4  52261.419159   3.136776  1.372190e-02
# baseprevcurr:delay        0.360033  0.180016      2     58.281082  13.887598  1.169012e-05
# group:baseprevcurr:delay  0.438330  0.109583      4     58.155717   8.453898  1.903805e-05

print(mumin.r_squaredGLMM(lmer))
# [[0.00869183 0.029561  ]] # first is only for fixed effects, last for fixed + random

# test significance of main effects
print(lmerTest.ranova(lmer))
#                                                     npar        logLik           AIC         LRT   Df    Pr(>Chisq)
# <none>                                                29  39280.715708 -78503.431415         NaN  NaN           NaN
# baseprevcurr:delay in (baseprevcurr:delay | sub...    20  39056.036564 -78072.073128  449.358287  9.0  3.932791e-91

# model for ctrl
lmer1 = ro.r('rlmer1 = lmer(error ~ baseprevcurr*delay + (baseprevcurr:delay|subject), data=rdat[rdat$group=="C",], control=optctrl)')
print(lmerTest.summary_lmerModLmerTest(lmer1))
print(lmerTest.anova_lmerModLmerTest(lmer1))        
#                       Sum Sq   Mean Sq  NumDF         DenDF    F value        Pr(>F)
# baseprevcurr        0.075912  0.075912      1     17.889944   6.594476  1.942022e-02
# delay               0.325869  0.162934      2  20150.681433  14.154201  7.198173e-07
# baseprevcurr:delay  0.619499  0.309750      2     16.931634  26.908123  5.528462e-06


lmer2 = ro.r('rlmer2 = lmer(error ~ baseprevcurr*delay + (baseprevcurr:delay|subject), data=rdat[rdat$group=="E",], control=optctrl)')
print(lmerTest.summary_lmerModLmerTest(lmer2))
print(lmerTest.anova_lmerModLmerTest(lmer2))        
#                       Sum Sq   Mean Sq  NumDF         DenDF    F value    Pr(>F)
# baseprevcurr        0.000452  0.000452      1     15.288903   0.032359  0.859605
# delay               0.279797  0.139898      2  16152.185007  10.022694  0.000045
# baseprevcurr:delay  0.141301  0.070650      2     22.752310   5.061577  0.015190
ro.r("isSingular(rlmer2)")                                                                                                                   
#array([0], dtype=int32)

lmer3 = ro.r('rlmer3 = lmer(error ~ baseprevcurr*delay + (baseprevcurr:delay|subject), data=rdat[rdat$group=="S",], control=optctrl)')
print(lmerTest.summary_lmerModLmerTest(lmer3))      ## singular
print(lmerTest.anova_lmerModLmerTest(lmer3))
#                       Sum Sq   Mean Sq  NumDF         DenDF   F value    Pr(>F)
# baseprevcurr        0.125123  0.125123      1     16.145664  9.074825  0.008197
# delay               0.075944  0.037972      2  15857.262922  2.754002  0.063703
# baseprevcurr:delay  0.036261  0.018131      2     15.734082  1.314971  0.296424
ro.r("isSingular(rlmer3)")                                                                                                                   
#array([1], dtype=int32)

lmer4 = ro.r('rlmer4 = lmer(error ~ baseprevcurr*group + (baseprevcurr|subject), data=rdat[rdat$delay==0,], control=optctrl)')
print(lmerTest.summary_lmerModLmerTest(lmer4))
print(lmerTest.anova_lmerModLmerTest(lmer4))
#                       Sum Sq   Mean Sq  NumDF      DenDF    F value    Pr(>F)
# baseprevcurr        0.068337  0.068337      1  51.624267  12.670087  0.000808
# group               0.006317  0.003159      2  49.639313   0.585627  0.560557
# baseprevcurr:group  0.004992  0.002496      2  51.579798   0.462791  0.632112


lmer5 = ro.r('rlmer5 = lmer(error ~ baseprevcurr*group + (baseprevcurr|subject), data=rdat[rdat$delay==60,], control=optctrl)')
print(lmerTest.summary_lmerModLmerTest(lmer5)  )
print(lmerTest.anova_lmerModLmerTest(lmer5))
#                       Sum Sq   Mean Sq  NumDF      DenDF   F value    Pr(>F)
# baseprevcurr        0.082114  0.082114      1  48.159455  6.190105  0.016359
# group               0.096256  0.048128      2  49.166282  3.628080  0.033909
# baseprevcurr:group  0.172905  0.086453      2  48.147290  6.517143  0.003127

lmer6 = ro.r('rlmer6 = lmer(error ~ baseprevcurr*group + (baseprevcurr|subject), data=rdat[rdat$delay==180,], control=optctrl)')
print(lmerTest.summary_lmerModLmerTest(lmer6))
print(lmerTest.anova_lmerModLmerTest(lmer6))
#                       Sum Sq   Mean Sq  NumDF      DenDF    F value    Pr(>F)
# baseprevcurr        0.069064  0.069064      1  50.222411   3.603173  0.063424
# group               0.037548  0.018774      2  50.073249   0.979466  0.382589
# baseprevcurr:group  0.588625  0.294313      2  50.157408  15.354803  0.000006


# fit bias-cleaned errors, make
dat['cleanerr']      = ro.r('rdat$res = residuals(rlmer)')
ro.globalenv['rdat'] = ro.conversion.py2rpy(dat) 

# run model in R to extract random intercepts and random slopes
random = ro.r('coef(rlmer)$subject')
cols   = random.columns.tolist()
cols   = cols[1:] + cols[:1]    # baseprevcrur*delay0 in last place
arand  = np.array(random[cols]) # for plotting

# # fixed coefficients for plotting
coef  = np.array(base.summary(lmer).rx2('coefficients'))[:,0]

# save stuff for figure 1
#np.save('../data/Fig_1/FIGURE1_coef_fix.npy', coef)
#np.save('../data/Fig_1/FIGURE1_coef_rand.npy', arand)
#dat.to_pickle('../data/Fig_1/FIGURE1_raw_dat.pkl')


##############################################################################
#                       SYMPTOMS AND RANDOM EFFECTS                          #
##############################################################################

# create symptoms dataset and append each subject's random slope
subs, subind    = np.unique(dat.subject, return_index=True)
symptoms        = dat.loc[subind,['subject','age','group','gender','dose','panssN',
                  'panssP','panssG','panssT','young','hamilton','gaf']].reset_index(drop=True)

symptoms['slope60'] = arand[:,13]
symptoms['slope180']= arand[:,12]

# save stuff for supplememtary figure 8
#symptoms.to_pickle('../data/Supplementary_Fig_8/SUPPLEMENTARY_FIGURE8_symptoms.pkl')


##############################################################################
#                               GROUP MODELS: ITI                            #
##############################################################################

groupID = dat.group[np.unique(dat.subject, return_index=True)[1]].values
ITIs    = dat.groupby(['subject', 'group'])['ITI'].median()
print(ITIs[groupID=="C"].mean()) # 2.71 +- 0.32
print(ITIs[groupID=="E"].mean()) # 2.91 +- 0.49
print(ITIs[groupID=="S"].mean()) # 3.03 +- 0.46

# Kruskal-Wallis
sps.kruskal(ITIs[groupID=="C"],ITIs[groupID=="E"],ITIs[groupID=="S"])
# KruskalResult(statistic=5.172170653839402, pvalue=0.07531429459605868)

lmer2 = ro.r('rlmer2 = lmer(error ~ group*baseprevcurr*delay + baseprevcurr:ITI + (baseprevcurr:delay|subject), data=rdat, control=optctrl)')
lmer3 = ro.r('rlmer3 = lmer(error ~ group*baseprevcurr*delay + baseprevcurr:ITI:delay + (baseprevcurr:delay|subject), data=rdat, control=optctrl)')

print(ro.r('anova(rlmer,rlmer2)'))
#           Df           AIC           BIC        logLik      deviance      Chisq  Chi Df  Pr(>Chisq)
# rlmer   29.0 -78673.100940 -78415.971067  39365.550470 -78731.100940        NaN     NaN         NaN
# rlmer2  30.0 -78686.877806 -78420.881385  39373.438903 -78746.877806  15.776866     1.0    0.000071

# print(ro.r('anova(rlmer2,rlmer3)'))
#           Df           AIC           BIC        logLik      deviance     Chisq  Chi Df  Pr(>Chisq)
# rlmer2  30.0 -78686.877806 -78420.881385  39373.438903 -78746.877806       NaN     NaN         NaN
# rlmer3  32.0 -78686.864281 -78403.134765  39375.432140 -78750.864281  3.986475     2.0    0.136254

print(lmerTest.summary_lmerModLmerTest(lmer2))
# baseprevcurr:ITI              8.014e-03  2.019e-03  7.504e+03   3.969 7.27e-05
print(lmerTest.anova_lmerModLmerTest(lmer2))
#                             Sum Sq   Mean Sq  NumDF         DenDF    F value        Pr(>F)
# group                     0.042816  0.021408      2     54.397023   1.651967  2.011425e-01
# baseprevcurr              0.115848  0.115848      1    664.918153   8.939457  2.893827e-03
# delay                     0.505609  0.252804      2  52263.110327  19.507778  3.396569e-09
# group:baseprevcurr        0.210736  0.105368      2     49.693365   8.130764  8.813225e-04
# group:delay               0.161919  0.040480      4  52260.068177   3.123647  1.403607e-02
# baseprevcurr:delay        0.363788  0.181894      2     58.335732  14.035938  1.055496e-05
# baseprevcurr:ITI          0.204183  0.204183      1   7503.506487  15.755905  7.273496e-05
# group:baseprevcurr:delay  0.438185  0.109546      4     58.211835   8.453197  1.901180e-05


##############################################################################
#               GROUP MODELS: MIXED EFFECTS FOR N-2 AND N+1                  #
##############################################################################

lmer1 = ro.r('rlmer1 = lmer(error ~ group*baseprevcurr*delay + group*baseprevcurr2*delay + (baseprevcurr:delay|subject), data=rdat, control=optctrl)') 
print(ro.r('anova(rlmer,rlmer1)'))
#           Df           AIC           BIC       logLik      deviance      Chisq  Chi Df  Pr(>Chisq)
# rlmer   29.0 -78673.100940 -78415.971067  39365.55047 -78731.100940        NaN     NaN         NaN
# rlmer1  38.0 -78677.218781 -78340.289981  39376.60939 -78753.218781  22.117841     9.0    0.008513

print(lmerTest.summary_lmerModLmerTest(lmer1))
print(lmerTest.anova_lmerModLmerTest(lmer1))
#                              Sum Sq   Mean Sq  NumDF         DenDF    F value        Pr(>F)
# group                      0.041359  0.020679      2     54.372954   1.595747  2.121058e-01
# baseprevcurr               0.016425  0.016425      1     49.281435   1.267445  2.656994e-01
# delay                      0.503239  0.251620      2  52255.921916  19.416441  3.721160e-09
# baseprevcurr2              0.001113  0.001113      1  52277.324820   0.085912  7.694407e-01
# group:baseprevcurr         0.251920  0.125960      2     49.262398   9.719792  2.766507e-04
# group:delay                0.165148  0.041287      4  52252.677306   3.185941  1.260523e-02
# baseprevcurr:delay         0.364185  0.182092      2     58.593166  14.051306  1.035518e-05
# group:baseprevcurr2        0.074100  0.037050      2  52275.891107   2.858998  5.733513e-02
# delay:baseprevcurr2        0.140851  0.070426      2  52269.026229   5.434442  4.366133e-03
# group:baseprevcurr:delay   0.435733  0.108933      4     58.466443   8.405915  1.992070e-05
# group:delay:baseprevcurr2  0.024334  0.006083      4  52268.034952   0.469434  7.582331e-01
ro.r("isSingular(rlmer1)")                                                                              
# array([0], dtype=int32)


lmer1a = ro.r('rlmer1a = lmer(error ~ group*baseprevcurr + group*baseprevcurr2 + (baseprevcurr|subject), data=rdat[rdat$delay==0,], control=optctrl)') 
print(lmerTest.anova_lmerModLmerTest(lmer1a))
#                        Sum Sq   Mean Sq  NumDF        DenDF    F value    Pr(>F)
# group                0.006337  0.003169      2    49.639034   0.588410  0.559035
# baseprevcurr         0.069077  0.069077      1    51.659856  12.827290  0.000755
# baseprevcurr2        0.082979  0.082979      1  8601.339999  15.409024  0.000087
# group:baseprevcurr   0.005321  0.002661      2    51.615051   0.494075  0.612991
# group:baseprevcurr2  0.001030  0.000515      2  8601.429748   0.095664  0.908770

lmer1b = ro.r('rlmer1b = lmer(error ~ group*baseprevcurr + group*baseprevcurr2 + (baseprevcurr|subject), data=rdat[rdat$delay==60,], control=optctrl)') 
print(lmerTest.anova_lmerModLmerTest(lmer1b))
#                        Sum Sq   Mean Sq  NumDF         DenDF   F value    Pr(>F)
# group                0.096219  0.048110      2     49.169662  3.627097  0.033937
# baseprevcurr         0.081620  0.081620      1     48.166410  6.153497  0.016660
# baseprevcurr2        0.000749  0.000749      1  34938.205374  0.056480  0.812149
# group:baseprevcurr   0.171800  0.085900      2     48.154231  6.476196  0.003229
# group:baseprevcurr2  0.089658  0.044829      2  34938.097962  3.379764  0.034067

lmer1c = ro.r('rlmer1c = lmer(error ~ group*baseprevcurr + group*baseprevcurr2 + (baseprevcurr|subject), data=rdat[rdat$delay==180,], control=optctrl)') 
print(lmerTest.anova_lmerModLmerTest(lmer1c))
#                        Sum Sq   Mean Sq  NumDF        DenDF    F value    Pr(>F)
# group                0.036341  0.018171      2    50.023554   0.948226  0.394280
# baseprevcurr         0.070401  0.070401      1    50.217158   3.673838  0.060972
# baseprevcurr2        0.049217  0.049217      1  8669.377643   2.568352  0.109058
# group:baseprevcurr   0.591447  0.295724      2    50.151618  15.432224  0.000006
# group:baseprevcurr2  0.059525  0.029762      2  8669.419289   1.553139  0.211642


lmer2 = ro.r('rlmer2 = lmer(error ~ group*baseprevcurr_1*delay + (baseprevcurr_1:delay|subject), data=rdat, control=optctrl)')
print(lmerTest.summary_lmerModLmerTest(lmer2))      ## singular
print(lmerTest.anova_lmerModLmerTest(lmer2))
#                               Sum Sq   Mean Sq  NumDF         DenDF    F value        Pr(>F)
# group                       0.041833  0.020917      2     54.363439   1.587550  2.137561e-01
# baseprevcurr_1              0.008329  0.008329      1     52.632474   0.632153  4.301375e-01
# delay                       0.498899  0.249450      2  52294.746866  18.932868  6.033027e-09
# group:baseprevcurr_1        0.041475  0.020738      2     52.590365   1.573957  2.168283e-01
# group:delay                 0.154846  0.038711      4  52293.642977   2.938141  1.929834e-02
# baseprevcurr_1:delay        0.056678  0.028339      2     88.026674   2.150894  1.224577e-01
# group:baseprevcurr_1:delay  0.054899  0.013725      4     87.969499   1.041691  3.904120e-01
ro.r("isSingular(rlmer2)")
# array([1], dtype=int32)

coef1  = np.array(base.summary(lmer1).rx2('coefficients'))[:,0]
coef2  = np.array(base.summary(lmer2).rx2('coefficients'))[:,0]

#np.save('../data/Supplementary_Fig_7/SUPPLEMENTARY_FIGURE7_coef1_fix.npy', coef1)
#np.save('../data/Supplementary_Fig_7/SUPPLEMENTARY_FIGURE7_coef2_fix.npy', coef2)

##############################################################################
#                               GROUP MODELS: DOSE                           #
##############################################################################

lmer2 = ro.r('rlmer2 = lmer(error ~ group*baseprevcurr*delay + baseprevcurr:dose + (baseprevcurr:delay|subject), data=rdat, control=optctrl)')
lmer3 = ro.r('rlmer3 = lmer(error ~ group*baseprevcurr*delay + baseprevcurr:dose:delay + (baseprevcurr:delay|subject), data=rdat, control=optctrl)')

print(ro.r('anova(rlmer,rlmer2)'))
#           Df           AIC           BIC        logLik      deviance     Chisq  Chi Df  Pr(>Chisq)
# rlmer   29.0 -78673.100940 -78415.971067  39365.550470 -78731.100940       NaN     NaN         NaN
# rlmer2  30.0 -78673.195243 -78407.198823  39366.597622 -78733.195243  2.094303     1.0    0.147849

print(ro.r('anova(rlmer,rlmer3)'))
#           Df           AIC           BIC        logLik      deviance     Chisq  Chi Df  Pr(>Chisq)
# rlmer   29.0 -78673.100940 -78415.971067  39365.550470 -78731.100940       NaN     NaN         NaN
# rlmer3  32.0 -78675.791034 -78392.061519  39369.895517 -78739.791034  8.690094     3.0    0.033708

print(lmerTest.summary_lmerModLmerTest(lmer3))
print(lmerTest.anova_lmerModLmerTest(lmer3))
#                             Sum Sq   Mean Sq  NumDF         DenDF    F value        Pr(>F)
# group                     0.042466  0.021233      2     54.380735   1.638098  2.037930e-01
# baseprevcurr              0.000006  0.000006      1     48.411390   0.000460  9.829701e-01
# delay                     0.507928  0.253964      2  52259.188442  19.592904  3.119599e-09
# group:baseprevcurr        0.101694  0.050847      2     48.326205   3.922779  2.638122e-02
# group:delay               0.162065  0.040516      4  52255.726066   3.125765  1.398491e-02
# baseprevcurr:delay        0.455855  0.227928      2     62.054182  17.584242  8.908956e-07
# group:baseprevcurr:delay  0.229883  0.057471      4     61.543844   4.433773  3.255749e-03
# baseprevcurr:delay:dose   0.118936  0.039645      3     60.131800   3.058571  3.496330e-02


##############################################################################
#                   GROUP MODELS: DOSE==0 AND GROUP!=SCHZ                    #
##############################################################################

lmer_d0 = ro.r('rlmer_d0 = lmer(error ~ group*baseprevcurr*delay + (baseprevcurr:delay|subject), data=rdat[(rdat$dose==0&rdat$group!="S"),], control=optctrl)')
print(lmerTest.summary_lmerModLmerTest(lmer_d0))
print(lmerTest.anova_lmerModLmerTest(lmer_d0))
#                             Sum Sq   Mean Sq  NumDF         DenDF    F value        Pr(>F)
# group                     0.032147  0.032147      1     31.972635   2.538336  1.209514e-01
# baseprevcurr              0.064663  0.064663      1     29.358268   5.105839  3.143551e-02
# delay                     0.355186  0.177593      2  32223.587014  14.022778  8.177748e-07
# group:baseprevcurr        0.016335  0.016335      1     29.358268   1.289849  2.652626e-01
# group:delay               0.060082  0.030041      2  32223.586929   2.372027  9.330768e-02
# baseprevcurr:delay        0.564921  0.282461      2     28.163595  22.303178  1.565244e-06
# group:baseprevcurr:delay  0.113745  0.056873      2     28.163595   4.490680  2.028629e-02

lmer_d = ro.r('rlmer_d = lmer(error ~ group*baseprevcurr*delay + (baseprevcurr:delay|subject), data=rdat[rdat$group!="S",], control=optctrl)')
print(lmerTest.summary_lmerModLmerTest(lmer_d))
print(lmerTest.anova_lmerModLmerTest(lmer_d))
#                             Sum Sq   Mean Sq  NumDF         DenDF    F value        Pr(>F)
# group                     0.028434  0.028434      1     36.295206   2.256249  1.417271e-01
# baseprevcurr              0.037472  0.037472      1     33.143250   2.973355  9.396280e-02
# delay                     0.482132  0.241066      2  36247.807925  19.128450  4.977377e-09
# group:baseprevcurr        0.049316  0.049316      1     33.143250   3.913222  5.626953e-02
# group:delay               0.119069  0.059535      2  36247.807856   4.724039  8.884707e-03
# baseprevcurr:delay        0.724482  0.362241      2     32.018913  28.743603  7.122624e-08
# group:baseprevcurr:delay  0.143585  0.071793      2     32.018913   5.696713  7.646174e-03

lmer_d01 = ro.r('rlmer_d01 = lmer(error ~ baseprevcurr*delay + (baseprevcurr:delay|subject), data=rdat[(rdat$dose==0&rdat$group=="C"),], control=optctrl)')
print(lmerTest.summary_lmerModLmerTest(lmer_d01))
print(lmerTest.anova_lmerModLmerTest(lmer_d01))
#                       Sum Sq   Mean Sq  NumDF         DenDF    F value        Pr(>F)
# baseprevcurr        0.075912  0.075912      1     17.889944   6.594476  1.942022e-02
# delay               0.325869  0.162934      2  20150.681433  14.154201  7.198173e-07
# baseprevcurr:delay  0.619499  0.309750      2     16.931634  26.908123  5.528462e-06
ro.r("isSingular(rlmer_d01)")  
# array([0], dtype=int32)

lmer_d02 = ro.r('rlmer_d02 = lmer(error ~ baseprevcurr*delay + (baseprevcurr:delay|subject), data=rdat[(rdat$dose==0&rdat$group=="E"),], control=optctrl)')
print(lmerTest.summary_lmerModLmerTest(lmer_d02)) 
print(lmerTest.anova_lmerModLmerTest(lmer_d02))
#                       Sum Sq   Mean Sq  NumDF         DenDF   F value    Pr(>F)
# baseprevcurr        0.009962  0.009962      1     11.557814  0.683572  0.425100
# delay               0.134816  0.067408      2  12151.760168  4.625439  0.009817
# baseprevcurr:delay  0.118507  0.059254      2     11.970359  4.065906  0.044928
ro.r("isSingular(rlmer_d02)")                                                                           
# array([1], dtype=int32)

lmer_d03 = ro.r('rlmer_d03 = lmer(error ~ baseprevcurr*group + (baseprevcurr|subject), data=rdat[(rdat$dose==0&rdat$group!="S"&rdat$delay==0),], control=optctrl)')
print(lmerTest.summary_lmerModLmerTest(lmer_d03))
print(lmerTest.anova_lmerModLmerTest(lmer_d03))
#                       Sum Sq   Mean Sq  NumDF      DenDF   F value    Pr(>F)
# baseprevcurr        0.019003  0.019003      1  30.071768  3.779363  0.061296
# group               0.017666  0.017666      1  30.160019  3.513425  0.070589
# baseprevcurr:group  0.000755  0.000755      1  30.071768  0.150141  0.701130

lmer_d04 = ro.r('rlmer_d04 = lmer(error ~ baseprevcurr*group + (baseprevcurr|subject), data=rdat[(rdat$dose==0&rdat$group!="S"&rdat$delay==60),], control=optctrl)')
print(lmerTest.summary_lmerModLmerTest(lmer_d04))
print(lmerTest.anova_lmerModLmerTest(lmer_d04))
#                       Sum Sq   Mean Sq  NumDF      DenDF   F value    Pr(>F)
# baseprevcurr        0.009467  0.009467      1  28.845169  0.733722  0.398740
# group               0.067214  0.067214      1  29.406587  5.209380  0.029877
# baseprevcurr:group  0.000002  0.000002      1  28.845169  0.000121  0.991287

lmer_d05 = ro.r('rlmer_d05 = lmer(error ~ baseprevcurr*group + (baseprevcurr|subject), data=rdat[(rdat$dose==0&rdat$group!="S"&rdat$delay==180),], control=optctrl)')
print(lmerTest.summary_lmerModLmerTest(lmer_d05))
print(lmerTest.anova_lmerModLmerTest(lmer_d05))
#                       Sum Sq   Mean Sq  NumDF      DenDF    F value    Pr(>F)
# baseprevcurr        0.482927  0.482927      1  29.623176  25.183316  0.000023
# group               0.006459  0.006459      1  29.707026   0.336842  0.566035
# baseprevcurr:group  0.111282  0.111282      1  29.623176   5.803026  0.022430

# run model in R to extract random intercepts and random slopes
random = ro.r('coef(rlmer_d0)$subject')
cols   = random.columns.tolist()
cols   = cols[1:] + cols[:1]            # baseprevcrur*delay0 in last place
arand = np.array(random[cols])

# # fixed and random coefficients for plotting - multiply covariate with condition mean
coef    = np.array(base.summary(lmer_d0).rx2('coefficients'))[:,0]

#np.save('../data/Supplementary_Fig_9/SUPPLEMENTARY_FIGURE9_coef_fix.npy', coef)
#np.save('../data/Supplementary_Fig_9/SUPPLEMENTARY_FIGURE9_coef_rand.npy', arand)


##############################################################################
#           GROUP MODELS: GROUP EFFECTS ON RESIDUALS FROM MEDICATION         #
##############################################################################

lm_cpz = ro.r('rlm_cpz = lm(error ~ dose + dose:baseprevcurr + dose:delay + dose:baseprevcurr:delay, data=rdat, control=optctrl)')
print(base.summary(lm_cpz))
# Coefficients:
#                              Estimate Std. Error t value Pr(>|t|)    
# (Intercept)                 1.422e-02  5.419e-04  26.230  < 2e-16 ***
# dose                       -1.207e-05  3.903e-06  -3.092  0.00199 ** 
# dose:baseprevcurr           2.064e-05  7.000e-06   2.949  0.00319 ** 
# dose:delay180               1.220e-05  5.466e-06   2.231  0.02568 *  
# dose:delay60                5.443e-06  4.306e-06   1.264  0.20621    
# dose:baseprevcurr:delay180  2.914e-05  9.890e-06   2.947  0.00321 ** 
# dose:baseprevcurr:delay60   2.100e-05  7.799e-06   2.692  0.00709 ** 

# calculate group differences in bias explained by dose: sum dose:pc and dose:delay180:pc coefs
# and multiply by avg dose per group
np.degrees(dat.groupby(["subject","group"]).mean().groupby("group").mean().dose * (2.064e-05+2.914e-05))
# C    0.000000
# E    0.075761
# S    1.056986

print(car.Anova(lm_cpz))
#                              Sum Sq       Df     F value        Pr(>F)
# dose                       0.186461      1.0   14.047962  1.783981e-04
# dose:baseprevcurr          2.603902      1.0  196.177968  1.715135e-44
# dose:delay                 0.062081      2.0    2.338604  9.647225e-02
# dose:baseprevcurr:delay    0.130237      2.0    4.906045  7.405103e-03
# Residuals                695.341076  52387.0         NaN           NaN

# write residuals as new column
dat['resid_cpz'] = ro.r('rdat$resid_cpz = residuals(rlm_cpz)')

# fit initital model on residuals
lmer_resid = ro.r('rlmer_resid = lmer(resid_cpz ~ group*baseprevcurr*delay + (baseprevcurr:delay|subject), data=rdat, control=optctrl)')
print(lmerTest.summary_lmerModLmerTest(lmer_resid))
#                                Estimate Std. Error         df t value Pr(>|t|)
# (Intercept)                  -8.802e-03  3.145e-03  1.093e+02  -2.798  0.00607
# groupE                        4.606e-03  4.666e-03  1.105e+02   0.987  0.32571
# groupS                        5.796e-03  4.634e-03  1.147e+02   1.251  0.21356
# baseprevcurr                  8.087e-03  5.014e-03  5.996e+01   1.613  0.11206
# delay180                      1.321e-02  2.793e-03  5.228e+04   4.730 2.26e-06
# delay60                       3.834e-03  2.212e-03  5.223e+04   1.733  0.08311
# groupE:baseprevcurr           2.084e-04  7.484e-03  6.202e+01   0.028  0.97787
# groupS:baseprevcurr           3.253e-04  7.426e-03  6.407e+01   0.044  0.96519
# groupE:delay180              -1.523e-03  4.171e-03  5.224e+04  -0.365  0.71497
# groupS:delay180              -1.115e-02  4.197e-03  5.227e+04  -2.656  0.00790
# groupE:delay60                6.933e-03  3.299e-03  5.223e+04   2.102  0.03558
# groupS:delay60               -4.118e-05  3.319e-03  5.222e+04  -0.012  0.99010
# baseprevcurr:delay180        -4.761e-02  6.060e-03  6.015e+01  -7.856 8.51e-11
# baseprevcurr:delay60         -1.138e-02  4.481e-03  8.768e+01  -2.540  0.01285
# groupE:baseprevcurr:delay180  2.815e-02  9.077e-03  6.239e+01   3.101  0.00289
# groupS:baseprevcurr:delay180  4.348e-02  9.046e-03  6.516e+01   4.807 9.41e-06
# groupE:baseprevcurr:delay60   5.353e-03  6.712e-03  9.166e+01   0.798  0.42719
# groupS:baseprevcurr:delay60   1.278e-02  6.697e-03  9.633e+01   1.908  0.05942

# calculate group differences in bias: sum groupS:pc and groupS:delay180:pc coefs
np.degrees(3.253e-04+4.348e-02)
# 2.5099

print(lmerTest.anova_lmerModLmerTest(lmer_resid))
#                             Sum Sq   Mean Sq  NumDF         DenDF    F value        Pr(>F)
# group                     0.035792  0.017896      2     54.612265   1.380697  2.600389e-01
# baseprevcurr              0.002891  0.002891      1     49.362276   0.223031  6.388214e-01
# delay                     0.376743  0.188372      2  52261.694993  14.532965  4.899684e-07
# group:baseprevcurr        0.091701  0.045850      2     49.341453   3.537376  3.667437e-02
# group:delay               0.231691  0.057923      4  52258.246914   4.468763  1.307043e-03
# baseprevcurr:delay        0.523888  0.261944      2     63.489227  20.209108  1.615650e-07
# group:baseprevcurr:delay  0.318132  0.079533      4     63.359117   6.136016  3.079973e-04
print(ro.r("isSingular(rlmer_resid)"))
# [0]

lmer_resid1 = ro.r('rlmer_resid1 = lmer(resid_cpz ~ group*baseprevcurr + (baseprevcurr|subject), data=rdat[rdat$delay==0,], control=optctrl)')
print(lmerTest.summary_lmerModLmerTest(lmer_resid1))
print(lmerTest.anova_lmerModLmerTest(lmer_resid1))
#                       Sum Sq   Mean Sq  NumDF      DenDF   F value    Pr(>F)
# group               0.018970  0.009485      2  49.684839  1.758556  0.182847
# baseprevcurr        0.036468  0.036468      1  51.513813  6.761365  0.012127
# group:baseprevcurr  0.000319  0.000160      2  51.470237  0.029610  0.970841

lmer_resid2 = ro.r('rlmer_resid2 = lmer(resid_cpz ~ group*baseprevcurr + (baseprevcurr|subject), data=rdat[rdat$delay==60,], control=optctrl)')
print(lmerTest.summary_lmerModLmerTest(lmer_resid2))
print(lmerTest.anova_lmerModLmerTest(lmer_resid2))
#                       Sum Sq   Mean Sq  NumDF      DenDF   F value    Pr(>F)
# group               0.099770  0.049885      2  49.211652  3.760527  0.030214
# baseprevcurr        0.010763  0.010763      1  48.128093  0.811380  0.372198
# group:baseprevcurr  0.036697  0.018348      2  48.115132  1.383171  0.260573

lmer_resid3 = ro.r('rlmer_resid3 = lmer(resid_cpz ~ group*baseprevcurr + (baseprevcurr|subject), data=rdat[rdat$delay==180,], control=optctrl)')
print(lmerTest.summary_lmerModLmerTest(lmer_resid3))
print(lmerTest.anova_lmerModLmerTest(lmer_resid3))
#                       Sum Sq   Mean Sq  NumDF      DenDF    F value    Pr(>F)
# group               0.037076  0.018538      2  50.056452   0.967160  0.387151
# baseprevcurr        0.238731  0.238731      1  50.501930  12.455138  0.000899
# group:baseprevcurr  0.343935  0.171968      2  50.428220   8.971947  0.000464

# run model in R to extract random intercepts and random slopes
random = ro.r('coef(rlmer_resid)$subject')
cols   = random.columns.tolist()
cols   = cols[1:] + cols[:1]            # baseprevcrur*delay0 in last place
arand  = np.array(random[cols])

# # fixed and random coefficients for plotting - multiply covariate with condition mean
coef    = np.array(base.summary(lmer_resid).rx2('coefficients'))[:,0]

#np.save('../data/Supplementary_Fig_9/SUPPLEMENTARY_FIGURE9_coef_fix_all.npy', coef)
#np.save('../data/Supplementary_Fig_9/SUPPLEMENTARY_FIGURE9_coef_rand_all.npy', arand)
#dat.to_pickle('../data/Supplementary_Fig_9/SUPPLEMENTARY_FIGURE9_raw_dat.pkl')


##############################################################################
#                               ANOVA PRECISION                              #
##############################################################################

# fit bias-cleaned errors
errstd = pd.DataFrame({'circstd': dat.groupby(['subject','delay',
         'group']).apply(lambda x: sps.circstd(x['cleanerr'].values))}).reset_index()

ro.globalenv['errstd'] = ro.conversion.py2rpy(errstd) 
m0 = ro.r('model = lm(circstd ~ delay*group, data=errstd)')
print(base.summary(m0))
print(stats.anova(m0))
# Coefficients:
#                  Estimate Std. Error t value Pr(>|t|)    
# (Intercept)      0.061047   0.006553   9.316  < 2e-16 ***
# delay180         0.068978   0.009268   7.443 7.65e-12 ***
# delay60          0.046034   0.009268   4.967 1.86e-06 ***
# groupE           0.005759   0.009692   0.594    0.553    
# groupS           0.012519   0.009536   1.313    0.191    
# delay180:groupE  0.002255   0.013707   0.165    0.870    
# delay60:groupE   0.002061   0.013707   0.150    0.881    
# delay180:groupS -0.004535   0.013486  -0.336    0.737    
# delay60:groupS  -0.003436   0.013486  -0.255    0.799    

#               Df    Sum Sq   Mean Sq    F value        Pr(>F)
# delay          2  0.125439  0.062719  76.865861  1.417914e-23
# group          2  0.002832  0.001416   1.735581  1.798905e-01
# delay:group    4  0.000224  0.000056   0.068570  9.913174e-01
# Residuals    147  0.119946  0.000816        NaN           NaN

# save stuff for figure 1
#errstd.to_pickle('../data/Fig_1/FIGURE1_errors.pkl')
