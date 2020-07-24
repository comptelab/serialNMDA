import numpy as np
import pandas as pd
import helpers as hf
import scipy.stats as sps
import scikits.bootstrap as boot
import statsmodels.formula.api as smf

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


##############################################################################
#                       LOAD AND PREPARE DATASET                             #
##############################################################################

# load first session data with neuropsychology attached
alldat      = pd.read_pickle('../data/behavior_retest.pkl')
par         = np.load('../data/DoG1_par_exp1_1000.npy')

# exclude outliers and first trials of each block (blocklength = 48 trials)
dat         = hf.filter_dat(alldat, rt=3, iti=5, raderr=5, err=1)
first       = dat.trial%48==1   
dat         = dat[~first].reset_index(drop=True)  # delete first trial of each block

# add DoG(prevcurr) to dataframe
# hyperparameter and DoG1 vs DoG3 from model_selection.py
dat['baseprevcurr']  = hf.dog1(par, dat.prevcurr.values)

 # create R dataframe for mixed models
ro.globalenv['rdat'] = ro.conversion.py2rpy(dat)


##############################################################################
#                               LINEAR MODEL                                 #
##############################################################################

m0 = ro.r('m0 = lm(error ~ group*session*baseprevcurr*delay, data=rdat)')
print(base.summary(m0))
print(car.Anova(m0))
#                                       Sum Sq       Df     F value        Pr(>F)
# group                               0.905944      1.0   70.549077  4.614864e-17
# session                             0.225927      1.0   17.593714  2.739979e-05
# baseprevcurr                        0.812494      1.0   63.271806  1.841072e-15
# delay                               0.740439      2.0   28.830308  3.068484e-13
# group:session                       0.002601      1.0    0.202512  6.527033e-01
# group:baseprevcurr                  1.460453      1.0  113.730623  1.602125e-26
# session:baseprevcurr                0.017893      1.0    1.393425  2.378336e-01
# group:delay                         0.083822      2.0    3.263746  3.825363e-02
# session:delay                       0.010595      2.0    0.412533  6.619740e-01
# baseprevcurr:delay                  1.625956      2.0   63.309455  3.487164e-28
# group:session:baseprevcurr          0.000844      1.0    0.065703  7.977005e-01
# group:session:delay                 0.066248      2.0    2.579475  7.582469e-02
# group:baseprevcurr:delay            0.166657      2.0    6.489098  1.521298e-03
# session:baseprevcurr:delay          0.065420      2.0    2.547250  7.830760e-02
# group:session:baseprevcurr:delay    0.021095      2.0    0.821367  4.398362e-01
# Residuals                         596.274458  46434.0         NaN           NaN


coefs = ro.r("m0$coef")
# (Intercept)                             ***
# groupE                                     
# sessionpre                              .  
# baseprevcurr                               
# delay180                                   
# delay60                                    
# groupE:sessionpre                          
# groupE:baseprevcurr                     *  
# sessionpre:baseprevcurr                    
# groupE:delay180                         ** 
# groupE:delay60                          ** 
# sessionpre:delay180                        
# sessionpre:delay60                         
# baseprevcurr:delay180                   ***
# baseprevcurr:delay60                    *  
# groupE:sessionpre:baseprevcurr             
# groupE:sessionpre:delay180              *  
# groupE:sessionpre:delay60               .  
# groupE:baseprevcurr:delay180            .  
# groupE:baseprevcurr:delay60                
# sessionpre:baseprevcurr:delay180           
# sessionpre:baseprevcurr:delay60            
# groupE:sessionpre:baseprevcurr:delay180    
# groupE:sessionpre:baseprevcurr:delay60     


m1 = ro.r('m1 = lm(error ~ session*baseprevcurr*delay, data=rdat[rdat$group=="E",])')
print(car.Anova(m1))
#                                 Sum Sq       Df    F value        Pr(>F)
# session                       0.171221      1.0  12.902647  3.286400e-04
# baseprevcurr                  0.000146      1.0   0.011007  9.164454e-01
# delay                         0.750507      2.0  28.277832  5.377841e-13
# session:baseprevcurr          0.008195      1.0   0.617584  4.319527e-01
# session:delay                 0.053210      2.0   2.004846  1.346991e-01
# baseprevcurr:delay            0.618457      2.0  23.302421  7.721606e-11
# session:baseprevcurr:delay    0.083996      2.0   3.164825  4.223559e-02
# Residuals                   399.739158  30123.0        NaN           NaN



m2 = ro.r('m2 = lm(error ~ session*baseprevcurr*delay, data=rdat[rdat$group=="C",])')
print(car.Anova(m2))
# Residuals                  196,535 16311                     
#                                 Sum Sq       Df     F value        Pr(>F)
# session                       0.057433      1.0    4.766524  2.903257e-02
# baseprevcurr                  2.259045      1.0  187.484328  1.932599e-42
# delay                         0.076666      2.0    3.181353  4.155520e-02
# session:baseprevcurr          0.010535      1.0    0.874318  3.497764e-01
# session:delay                 0.024936      2.0    1.034754  3.553370e-01
# baseprevcurr:delay            1.172463      2.0   48.652934  8.571878e-22
# session:baseprevcurr:delay    0.002519      2.0    0.104537  9.007423e-01
# Residuals                   196.535300  16311.0         NaN           NaN


m1 = ro.r('m1 = lm(error ~ session*baseprevcurr*group, data=rdat[rdat$delay==0,])')
print(base.summary(m1))
print(car.Anova(m1))
#                                Sum Sq      Df    F value    Pr(>F)
# session                      0.009865     1.0   2.230102  0.135386
# baseprevcurr                 0.065107     1.0  14.718316  0.000126
# group                        0.016402     1.0   3.707966  0.054190
# session:baseprevcurr         0.000289     1.0   0.065426  0.798124
# session:group                0.032544     1.0   7.356883  0.006696
# baseprevcurr:group           0.066049     1.0  14.931130  0.000112
# session:baseprevcurr:group   0.004663     1.0   1.054200  0.304574
# Residuals                   33.959557  7677.0        NaN       NaN

                
m2 = ro.r('m2 = lm(error ~ session*baseprevcurr*group, data=rdat[rdat$delay==60,])')
print(base.summary(m2))
print(car.Anova(m2))
#                                 Sum Sq       Df    F value        Pr(>F)
# session                       0.177595      1.0  13.322317  2.626965e-04
# baseprevcurr                  0.250588      1.0  18.797888  1.457858e-05
# group                         0.780781      1.0  58.570313  2.018375e-14
# session:baseprevcurr          0.000110      1.0   0.008231  9.277132e-01
# session:group                 0.003790      1.0   0.284286  5.939093e-01
# baseprevcurr:group            0.880625      1.0  66.060137  4.534931e-16
# session:baseprevcurr:group    0.003628      1.0   0.272146  6.018996e-01
# Residuals                   413.316957  31005.0        NaN           NaN



m3 = ro.r('m3 = lm(error ~ session*baseprevcurr*group, data=rdat[rdat$delay==180,])')
print(base.summary(m3))
print(car.Anova(m3))
#                                 Sum Sq      Df     F value        Pr(>F)
# session                       0.049136     1.0    2.556411  1.098885e-01
# baseprevcurr                  2.110578     1.0  109.808212  1.597892e-25
# group                         0.187980     1.0    9.780151  1.770589e-03
# session:baseprevcurr          0.082908     1.0    4.313485  3.784427e-02
# session:group                 0.032474     1.0    1.689523  1.937025e-01
# baseprevcurr:group            0.680257     1.0   35.392102  2.812888e-09
# session:baseprevcurr:group    0.013647     1.0    0.710042  3.994557e-01
# Residuals                   148.997944  7752.0         NaN           NaN


m1 = ro.r('m1 = lm(error ~ session*baseprevcurr, data=rdat[rdat$group=="E"&rdat$delay==0,])')
print(car.Anova(m1))
#                          Sum Sq      Df    F value        Pr(>F)
# session                0.000643     1.0   0.140832  7.074709e-01
# baseprevcurr           0.128460     1.0  28.129704  1.183076e-07
# session:baseprevcurr   0.000749     1.0   0.163947  6.855653e-01
# Residuals             22.979648  5032.0        NaN           NaN


m2 = ro.r('m2 = lm(error ~ session*baseprevcurr, data=rdat[rdat$group=="E"&rdat$delay==60,])')
print(car.Anova(m2))
#                           Sum Sq       Df    F value    Pr(>F)
# session                 0.141016      1.0  10.222611  0.001389
# baseprevcurr            0.023040      1.0   1.670191  0.196248
# session:baseprevcurr    0.000739      1.0   0.053604  0.816910
# Residuals             276.732139  20061.0        NaN       NaN


m3 = ro.r('m3 = lm(error ~ session*baseprevcurr, data=rdat[rdat$group=="E"&rdat$delay==180,])')
print(car.Anova(m3))
#                           Sum Sq      Df    F value    Pr(>F)
# session                 0.082983     1.0   4.172911  0.041127
# baseprevcurr            0.467377     1.0  23.502639  0.000001
# session:baseprevcurr    0.090703     1.0   4.561125  0.032754
# Residuals             100.027371  5030.0        NaN       NaN



##############################################################################
#                       GROUP MODELS: SAVE COEFS                             #
##############################################################################

#np.save('../data/Supplementary_Fig_10/SUPPLEMENTARY_FIGURE10_coefs.npy', coefs)
#dat.to_pickle('../data/Supplementary_Fig_10/SUPPLEMENTARY_FIGURE10_dat.pkl')


##############################################################################
#               INDIVIDUAL ESTIMATES: CORRELATE WITH SYMPTOMS                #
##############################################################################

# lmer = ro.r('rlmer = lmer(error ~ delay*group*baseprevcurr + (baseprevcurr:delay|subject), data=rdat[rdat$group!="S"&rdat$session=="post",])')
# print(lmerTest.summary_lmerModLmerTest(lmer))
# print(lmerTest.anova_lmerModLmerTest(lmer))
# fails to converge

def linmod_main(dat):
    m = smf.ols("error ~ baseprevcurr", data=dat)
    return - m.fit().params['baseprevcurr']

pars = dat.groupby(['subject','session','group','delay']).apply(linmod_main).reset_index()
d_bias = pars[((pars.delay=='180')&(pars.group=='E'))].reset_index(drop=True)
d_symps = dat[dat.group=='E'].groupby(['subject','session']).mean().reset_index()
d_symps['bias'] = d_bias[0]

# correlate sympts for post session
print(sps.pearsonr(d_symps[d_symps.session=='post'].bias, d_symps[d_symps.session=='post'].panssP))
print(sps.pearsonr(d_symps[d_symps.session=='post'].bias, d_symps[d_symps.session=='post'].panssN))
print(sps.pearsonr(d_symps[d_symps.session=='post'].bias, d_symps[d_symps.session=='post'].panssG))
print(sps.pearsonr(d_symps[d_symps.session=='post'].bias, d_symps[d_symps.session=='post'].young))
print(sps.pearsonr(d_symps[d_symps.session=='post'].bias, d_symps[d_symps.session=='post'].hamilton))
print(sps.pearsonr(d_symps[d_symps.session=='post'].bias, d_symps[d_symps.session=='post'].gaf)) 

# (-0.6950922562481033, 0.0057877886523887605)
# (-0.19000942480544647, 0.5152822225942673)
# (-0.62088294841492, 0.01780863518370692)
# (-0.22031976809191833, 0.4491231462374078)
# (-0.26612242911627115, 0.3577601334265613)
# (0.41927328891365856, 0.13562235374752266)


# correlate sympts for post session
print(sps.pearsonr(d_symps[d_symps.session=='pre'].bias, d_symps[d_symps.session=='pre'].panssP))
print(sps.pearsonr(d_symps[d_symps.session=='pre'].bias, d_symps[d_symps.session=='pre'].panssN))
print(sps.pearsonr(d_symps[d_symps.session=='pre'].bias, d_symps[d_symps.session=='pre'].panssG))
print(sps.pearsonr(d_symps[d_symps.session=='pre'].bias, d_symps[d_symps.session=='pre'].young))
print(sps.pearsonr(d_symps[d_symps.session=='pre'].bias, d_symps[d_symps.session=='pre'].hamilton))
print(sps.pearsonr(d_symps[d_symps.session=='pre'].bias, d_symps[d_symps.session=='pre'].gaf))

# (-0.37541927602061115, 0.1859223883136987)
# (-0.0731180655964544, 0.8038195048223588)
# (-0.019620770160919537, 0.9469200255472001)
# (0.4127944115588086, 0.14240354147691126)
# (0.27874300675800745, 0.3345273480927079)
# (-0.3897640830725153, 0.168321431363652)



# parametric C.I.
from rpy2.robjects.packages import importr 
psychom = importr("psychometric") 

corr = np.float(sps.pearsonr(d_symps[d_symps.session=='post'].bias, d_symps[d_symps.session=='post'].panssP)[0])
print(psychom.CIr(r=corr, n=len(d_symps[d_symps.session=='post'].bias), level=0.95))
# [-0.89543413 -0.26063561]

boot.ci((d_symps[d_symps.session=='post'].bias, d_symps[d_symps.session=='post'].panssP), sps.pearsonr)
# array([[-8.76344443e-01,  2.32234397e-05],
#        [-4.19492437e-01,  1.14862316e-01]])

corr = np.float(sps.pearsonr(d_symps[d_symps.session=='post'].bias, d_symps[d_symps.session=='post'].panssG)[0])
print(psychom.CIr(r=corr, n=len(d_symps[d_symps.session=='post'].bias), level=0.95))
# [-0.8661339  -0.13466625]

boot.ci((d_symps[d_symps.session=='post'].bias, d_symps[d_symps.session=='post'].panssG), sps.pearsonr)
# array([[-0.77428576,  0.00098805],
#        [-0.30565438,  0.23129379]])

corr = np.float(sps.pearsonr(d_symps[d_symps.session=='pre'].bias, d_symps[d_symps.session=='pre'].panssP)[0])
print(psychom.CIr(r=corr, n=len(d_symps[d_symps.session=='pre'].bias), level=0.95))
# [-0.75550909  0.19375405]

boot.ci((d_symps[d_symps.session=='pre'].bias, d_symps[d_symps.session=='pre'].panssP), sps.pearsonr)
# array([[-0.68810675,  0.00508161],
#        [ 0.265058  ,  0.93543675]])

corr = np.float(sps.pearsonr(d_symps[d_symps.session=='pre'].bias, d_symps[d_symps.session=='pre'].panssG)[0])
print(psychom.CIr(r=corr, n=len(d_symps[d_symps.session=='pre'].bias), level=0.95))
# [-0.54453149  0.51633396]

boot.ci((d_symps[d_symps.session=='pre'].bias, d_symps[d_symps.session=='pre'].panssG), sps.pearsonr)
# array([[-0.60071312,  0.90026091],
#        [ 0.52419035,  0.99998231]])
