# FIGURE 1
# SERIAL BIAS GROUP DIFFERENCES AND RANDOM EFFECTS

# author: heike stein
# last mod: 23/04/20

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import helpers as hf
import scikits.bootstrap as boot
import scipy.stats as sps

##############################################################################
#                           PLOTTING PARAMETERS                              #
##############################################################################

np.set_printoptions(precision=4)
sns.set_context("talk", font_scale=1)
sns.set_style("ticks")

uglybrown   = sns.xkcd_rgb["ugly brown"]
darkmauve   = sns.xkcd_rgb["dark mauve"]
darkorange  = sns.xkcd_rgb["dark orange"]


##############################################################################
#                   ADDITIONAL FUNCTIONS - PLOTTING ETC.                     #
##############################################################################

def sig_labels(x1,x2,y,ysize,p):
    if p>=0.01: text=r'$p=%.2f$' %p
    elif p>=0.001: text=r'$p=%.3f$' %p 
    else: text=r'$p=%.1e$' %p 
    plt.plot([x1,x2], [y,y], 'k', lw=1)
    plt.plot([x1,x1], [y-ysize,y], 'k', lw=1)
    plt.plot([x2,x2], [y-ysize,y], 'k', lw=1)
    plt.text(x1, y+.0025, text, fontsize=12)

def plot_ci(x,mean,ci,capsize):
    plt.plot([x,x],ci, 'k-', lw=2)
    plt.plot([x-capsize, x+capsize], [ci[0],ci[0]], 'k-', lw=2)
    plt.plot([x-capsize, x+capsize], [ci[1],ci[1]], 'k-', lw=2)
    plt.plot([x-capsize/2, x+capsize/2], [mean,mean], 'k-', lw=2)

def cohen_d(x1,x2):
    s = np.sqrt( ((len(x1)-1)*np.var(x1) + (len(x2)-1)*np.var(x2))/(len(x1)+len(x2)-2) )
    return (np.mean(x1)-np.mean(x2))/s


##############################################################################
#                   LOAD DATA AND CREATE IDENTIFIERS                         #
##############################################################################

coef    = np.load('../data/Fig_1/FIGURE1_coef_fix.npy')
arand   = np.load('../data/Fig_1/FIGURE1_coef_rand.npy')
dat     = pd.read_pickle('../data/Fig_1/FIGURE1_raw_dat.pkl')
dat1    = pd.read_pickle('../data/Fig_1/FIGURE1_errors.pkl')

# group identifiers
ctrl    = np.array([s[0] for s in list(dat.subject.unique())]) == 'C'
enc     = np.array([s[0] for s in list(dat.subject.unique())]) == 'E'
schz    = np.array([s[0] for s in list(dat.subject.unique())]) == 'S'

dat['prevcurr'] = hf.circdist(dat.serial.values,dat.target.values)


##############################################################################
#                       SMOOTH SERIAL BIAS CURVES                            #
##############################################################################

# serial bias parameters for sliding average
window      = np.pi/3
step        = np.pi/30

sb = []; se = []
for group in np.unique(dat.group):
    for delay in np.unique(dat.delay):
        cond = ((dat.group == group) & (dat.delay == delay))
        sb.append(hf.folded_bias(np.array(dat.prevcurr[cond]),np.array(dat.error[cond]),window,step)[0])
        se.append(hf.folded_bias(np.array(dat.prevcurr[cond]),np.array(dat.error[cond]),window,step)[1])

sb_ctrl = np.array(sb[:3]);     se_ctrl = np.array(se[:3])
sb_enc  = np.array(sb[3:6]);    se_enc  = np.array(se[3:6])
sb_schz = np.array(sb[6:]);     se_schz = np.array(se[6:])


##############################################################################
#                               PRECISION                                    #
##############################################################################

prec0     = dat1.circstd[dat1.delay=='0'].values
prec60    = dat1.circstd[dat1.delay=='60'].values
prec180   = dat1.circstd[dat1.delay=='180'].values


##############################################################################
#                           GET FIXED AND RANDOM EFFECTS                     #
##############################################################################

# axes and basis function
xfits   = np.arange(-1.5*np.pi, 1.5*np.pi, .01)
dogx    = hf.dog1(-.8,xfits)
xxx     = np.arange(0, np.pi, step)


# fixed effects
#         pc             group*pc       delay*pc        group*delay*pc
ctrl0   = coef[3]*dogx                                                
ctrl60  = coef[3]*dogx                + coef[13]*dogx                 
ctrl180 = coef[3]*dogx                + coef[12]*dogx                 
enc0    = coef[3]*dogx + coef[6]*dogx                                 
enc60   = coef[3]*dogx + coef[6]*dogx + coef[13]*dogx + coef[16]*dogx 
enc180  = coef[3]*dogx + coef[6]*dogx + coef[12]*dogx + coef[14]*dogx 
schz0   = coef[3]*dogx + coef[7]*dogx                                 
schz60  = coef[3]*dogx + coef[7]*dogx + coef[13]*dogx + coef[17]*dogx 
schz180 = coef[3]*dogx + coef[7]*dogx + coef[12]*dogx + coef[15]*dogx 


# random effects
#              pc            group*pc     delay*pc      group*delay*pc
rctrl0      = -(arand[:,3]              + arand[:,-1]               )   [ctrl]
rctrl60     = -(arand[:,3]              + arand[:,13]               )   [ctrl]
rctrl180    = -(arand[:,3]              + arand[:,12]               )   [ctrl]
renc0       = -(arand[:,3] + arand[:,6] + arand[:,-1]               )   [enc]
renc60      = -(arand[:,3] + arand[:,6] + arand[:,13] + arand[:,16] )   [enc]
renc180     = -(arand[:,3] + arand[:,6] + arand[:,12] + arand[:,14] )   [enc]
rschz0      = -(arand[:,3] + arand[:,7] + arand[:,-1]               )   [schz]
rschz60     = -(arand[:,3] + arand[:,7] + arand[:,13] + arand[:,17] )   [schz]
rschz180    = -(arand[:,3] + arand[:,7] + arand[:,12] + arand[:,15] )   [schz]


##############################################################################
#                                    FIGURE                                  #
##############################################################################

fig1 =plt.figure(figsize=[12,12])
ax0=plt.subplot(3,3,1)
plt.ylabel(r'circular s.d. of errors ($^\circ$)')
plt.xticks([])
plt.yticks([])
plt.plot(np.arange(-1,3,.01), hf.normgauss(np.arange(-2,2,.01),.4), darkorange)
plt.plot(xfits+3,np.zeros(len(xfits)),'k--')
plt.ylim(-.3,4)
sns.despine(ax=ax0, left=True, bottom=True)

ax1=plt.subplot(3,3,2)
plt.xticks([])
plt.yticks([])
plt.plot(np.arange(-1.8,2.2,.01), hf.normgauss(np.arange(-2,2,.01),.4), darkorange)
plt.plot(xfits+3,np.zeros(len(xfits)),'k--')
plt.ylim(-.3,4)
sns.despine(ax=ax1, left=True, bottom=True)

ax2=plt.subplot(3,3,3)
plt.plot(np.zeros(sum(ctrl))+1+np.random.randn(sum(ctrl))/8,prec0[ctrl],'o',color='k', alpha=.3, label='ctrl')
plt.plot(np.zeros(sum(enc))+2+np.random.randn(sum(enc))/8,prec0[enc],'o',color=uglybrown, alpha=.5, label='enc')
plt.plot(np.zeros(sum(schz))+3+np.random.randn(sum(schz))/8,prec0[schz],'o',color=darkmauve, alpha=.5, label='schz')
plot_ci(1, sps.circmean(prec0[ctrl]), sps.circmean(prec0[ctrl])+[sps.circstd(prec0[ctrl])/np.sqrt(sum(ctrl)),
    -sps.circstd(prec0[ctrl])/np.sqrt(sum(ctrl))], capsize=.4)
plot_ci(2, sps.circmean(prec0[enc]), sps.circmean(prec0[enc])+[sps.circstd(prec0[enc])/np.sqrt(sum(enc)),
    -sps.circstd(prec0[enc])/np.sqrt(sum(enc))], capsize=.4)
plot_ci(3, sps.circmean(prec0[schz]), sps.circmean(prec0[schz])+[sps.circstd(prec0[schz])/np.sqrt(sum(schz)),
    -sps.circstd(prec0[schz])/np.sqrt(sum(schz))], capsize=.4)

plt.plot(np.zeros(sum(ctrl))+5+np.random.randn(sum(ctrl))/8,prec60[ctrl],'o',color='k', alpha=.3)
plt.plot(np.zeros(sum(enc))+6+np.random.randn(sum(enc))/8,prec60[enc],'o',color=uglybrown, alpha=.5)
plt.plot(np.zeros(sum(schz))+7+np.random.randn(sum(schz))/8,prec60[schz],'o',color=darkmauve, alpha=.5)
plot_ci(5, sps.circmean(prec60[ctrl]), sps.circmean(prec60[ctrl])+[sps.circstd(prec60[ctrl])/np.sqrt(sum(ctrl)),
    -sps.circstd(prec60[ctrl])/np.sqrt(sum(ctrl))], capsize=.4)
plot_ci(6, sps.circmean(prec60[enc]), sps.circmean(prec60[enc])+[sps.circstd(prec60[enc])/np.sqrt(sum(enc)),
    -sps.circstd(prec60[enc])/np.sqrt(sum(enc))], capsize=.4)
plot_ci(7, sps.circmean(prec60[schz]), sps.circmean(prec60[schz])+[sps.circstd(prec60[schz])/np.sqrt(sum(schz)),
    -sps.circstd(prec60[schz])/np.sqrt(sum(schz))], capsize=.4)

plt.plot(np.zeros(sum(ctrl))+9+np.random.randn(sum(ctrl))/8,prec180[ctrl],'o',color='k', alpha=.3)
plt.plot(np.zeros(sum(enc))+10+np.random.randn(sum(enc))/8,prec180[enc],'o',color=uglybrown, alpha=.5)
plt.plot(np.zeros(sum(schz))+11+np.random.randn(sum(schz))/8,prec180[schz],'o',color=darkmauve, alpha=.5)
plot_ci(9, sps.circmean(prec180[ctrl]), sps.circmean(prec180[ctrl])+[sps.circstd(prec180[ctrl])/np.sqrt(sum(ctrl)),
    -sps.circstd(prec180[ctrl])/np.sqrt(sum(ctrl))], capsize=.4)
plot_ci(10, sps.circmean(prec180[enc]), sps.circmean(prec180[enc])+[sps.circstd(prec180[enc])/np.sqrt(sum(enc)),
    -sps.circstd(prec180[enc])/np.sqrt(sum(enc))], capsize=.4)
plot_ci(11, sps.circmean(prec180[schz]), sps.circmean(prec180[schz])+[sps.circstd(prec180[schz])/np.sqrt(sum(schz)),
    -sps.circstd(prec180[schz])/np.sqrt(sum(schz))], capsize=.4)
plt.plot([0,12],[0,0],'k--',alpha=.3)
plt.xticks([2,6,10],['0 s', '1 s', '3 s'])
plt.yticks(np.deg2rad([0,5,10,15]),[0,5,10,15])
plt.ylim([np.deg2rad(0),np.deg2rad(15)])
plt.xlabel('delay length')
plt.ylabel(r'circular SD of errors ($^\circ$)')
ax2.get_yaxis().set_tick_params(direction='in')
ax2.get_xaxis().set_tick_params(direction='in')
#plt.legend()
sns.despine()


ax3=plt.subplot(3,3,4)
plt.plot(xfits, ctrl0, 'k')
plt.plot(xfits, enc0, uglybrown)
plt.plot(xfits, schz0, darkmauve)
plt.plot(xxx,sb_ctrl[0], '--', color='k')
plt.plot(xxx,sb_enc[0], '--', color=uglybrown)
plt.plot(xxx,sb_schz[0], '--', color=darkmauve)
plt.fill_between(xxx,sb_ctrl[0]+se_ctrl[0], sb_ctrl[0]-se_ctrl[0], color='k', alpha=.3, label='ctrl, n=19')
plt.fill_between(xxx,sb_enc[0]+se_enc[0], sb_enc[0]-se_enc[0], color=uglybrown, alpha=.3, label='enc, n=16')
plt.fill_between(xxx,sb_schz[0]+se_schz[0], sb_schz[0]-se_schz[0], color=darkmauve, alpha=.3, label='schz, n=17')
plt.plot(xfits,np.zeros(len(xfits)), 'k--', alpha=.3)
plt.ylim([-.04,.04])
plt.xlim([-np.pi,np.pi])
plt.xlim([0,np.pi])
plt.xticks([0,np.pi/2,np.pi],[0,90,180])
plt.yticks([-np.deg2rad(3),0,np.deg2rad(3)],[-3,0,3])
plt.ylabel(r'error in current trial $\theta^{e}$ ($^\circ$)')
plt.title('0 seconds delay')
ax3.get_yaxis().set_tick_params(direction='in')
ax3.get_xaxis().set_tick_params(direction='in')
plt.legend()
sns.despine()

ax4=plt.subplot(3,3,5)
plt.plot(xfits, ctrl60, 'k')
plt.plot(xfits, enc60, uglybrown)
plt.plot(xfits, schz60, darkmauve)
plt.plot(xxx,sb_ctrl[2], '--', color='k')
plt.plot(xxx,sb_enc[2], '--', color=uglybrown)
plt.plot(xxx,sb_schz[2], '--', color=darkmauve)
plt.fill_between(xxx,sb_ctrl[2]+se_ctrl[2], sb_ctrl[2]-se_ctrl[2], color='k', alpha=.3)
plt.fill_between(xxx,sb_enc[2]+se_enc[2], sb_enc[2]-se_enc[2], color=uglybrown, alpha=.3)
plt.fill_between(xxx,sb_schz[2]+se_schz[2], sb_schz[2]-se_schz[2], color=darkmauve, alpha=.3)
plt.plot(xfits,np.zeros(len(xfits)), 'k--', alpha=.3)
plt.ylim([-.04,.04])
plt.xlim([-np.pi,np.pi])
plt.xlim([0,np.pi])
plt.xticks([0,np.pi/2,np.pi],[0,90,180])
plt.yticks([-np.deg2rad(3),0,np.deg2rad(3)],[-3,0,3])
plt.xlabel(r'absolute distance $|\theta^{d}|$ ($^\circ$)')
plt.title('1 second delay')
ax4.get_yaxis().set_tick_params(direction='in')
ax4.get_xaxis().set_tick_params(direction='in')
sns.despine()

ax5=plt.subplot(3,3,6)
plt.plot(xfits, ctrl180, 'k')
plt.plot(xfits, enc180, uglybrown)
plt.plot(xfits, schz180, darkmauve)
plt.plot(xxx,sb_ctrl[1], '--', color='k')
plt.plot(xxx,sb_enc[1], '--', color=uglybrown)
plt.plot(xxx,sb_schz[1], '--', color=darkmauve)
plt.fill_between(xxx,sb_ctrl[1]+se_ctrl[1], sb_ctrl[1]-se_ctrl[1], color='k', alpha=.3)
plt.fill_between(xxx,sb_enc[1]+se_enc[1], sb_enc[1]-se_enc[1], color=uglybrown, alpha=.3)
plt.fill_between(xxx,sb_schz[1]+se_schz[1], sb_schz[1]-se_schz[1], color=darkmauve, alpha=.3)
plt.plot(xfits,np.zeros(len(xfits)), 'k--', alpha=.3)
plt.ylim([-.04,.04])
plt.xlim([-np.pi,np.pi])
plt.xlim([0,np.pi])
plt.xticks([0,np.pi/2,np.pi],[0,90,180])
plt.yticks([-np.deg2rad(3),0,np.deg2rad(3)],[-3,0,3])
plt.title('3 seconds delay')
ax5.get_yaxis().set_tick_params(direction='in')
ax5.get_xaxis().set_tick_params(direction='in')
sns.despine()


##############################################################################
#                                    FIGURE                                  #
##############################################################################


ax6=plt.subplot(3,3,7)
plt.plot(np.ones(sum(ctrl))+np.random.randn(sum(ctrl))/8,rctrl0,'o', alpha=.3, color='k')
plt.plot(np.ones(sum(enc))+1+np.random.randn(sum(enc))/8,renc0,'o', alpha=.5, color=uglybrown)
plt.plot(np.ones(sum(schz))+2+np.random.randn(sum(schz))/8,rschz0,'o', alpha=.5, color=darkmauve)
plot_ci(1, np.mean(rctrl0), boot.ci(rctrl0), capsize=.3)
plot_ci(2, np.mean(renc0), boot.ci(renc0), capsize=.3)
plot_ci(3, np.mean(rschz0), boot.ci(rschz0), capsize=.3)
sig_labels(0.8,1.9,np.deg2rad(6),np.deg2rad(.25), sps.ttest_ind(rctrl0,renc0)[1])
sig_labels(2.1,3.2,np.deg2rad(6),np.deg2rad(.25), sps.ttest_ind(renc0,rschz0)[1])
sig_labels(0.8,3.2,np.deg2rad(7.5),np.deg2rad(.25), sps.ttest_ind(rctrl0,rschz0)[1])
plt.xlim([0,4])
plt.ylim(np.deg2rad([-8,8]))
plt.yticks(np.deg2rad([-8,0,8]),[-8,0,8])
plt.xticks([1,2,3],['ctrl','enc','schz'])
plt.ylabel(r'bias strength ($^\circ$)')
plt.plot([0,4],np.zeros(2),'k--', alpha=.3)
ax6.get_yaxis().set_tick_params(direction='in')
ax6.get_xaxis().set_tick_params(direction='in')
sns.despine()

ax7=plt.subplot(3,3,8)
plt.plot(np.ones(sum(ctrl))+np.random.randn(sum(ctrl))/8,rctrl60,'o', alpha=.3, color='k')
plt.plot(np.ones(sum(enc))+1+np.random.randn(sum(enc))/8,renc60,'o', alpha=.5, color=uglybrown)
plt.plot(np.ones(sum(schz))+2+np.random.randn(sum(schz))/8,rschz60,'o', alpha=.5, color=darkmauve)
plot_ci(1, np.mean(rctrl60), boot.ci(rctrl60), capsize=.3)
plot_ci(2, np.mean(renc60), boot.ci(renc60), capsize=.3)
plot_ci(3, np.mean(rschz60), boot.ci(rschz60), capsize=.3)
sig_labels(0.8,1.9,np.deg2rad(6),np.deg2rad(.25), sps.ttest_ind(rctrl60,renc60)[1])
sig_labels(2.1,3.2,np.deg2rad(6),np.deg2rad(.25), sps.ttest_ind(renc60,rschz60)[1])
sig_labels(0.8,3.2,np.deg2rad(7.5),np.deg2rad(.25), sps.ttest_ind(rctrl60,rschz60)[1])
plt.xlim([0,4])
plt.ylim(np.deg2rad([-8,8]))
plt.yticks(np.deg2rad([-8,0,8]),[-8,0,8])
plt.xticks([1,2,3],['ctrl','enc','schz'])
plt.plot([0,4],np.zeros(2),'k--', alpha=.3)
ax7.get_yaxis().set_tick_params(direction='in')
ax7.get_xaxis().set_tick_params(direction='in')
sns.despine()

ax8=plt.subplot(3,3,9)
plt.plot(np.ones(sum(ctrl))+np.random.randn(sum(ctrl))/8,rctrl180,'o', alpha=.3, color='k')
plt.plot(np.ones(sum(enc))+1+np.random.randn(sum(enc))/8,renc180,'o', alpha=.5, color=uglybrown)
plt.plot(np.ones(sum(schz))+2+np.random.randn(sum(schz))/8,rschz180,'o', alpha=.5, color=darkmauve)
plot_ci(1, np.mean(rctrl180), boot.ci(rctrl180), capsize=.3)
plot_ci(2, np.mean(renc180), boot.ci(renc180), capsize=.3)
plot_ci(3, np.mean(rschz180), boot.ci(rschz180), capsize=.3)
sig_labels(0.8,1.9,np.deg2rad(6),np.deg2rad(.25), sps.ttest_ind(rctrl180,renc180)[1])
sig_labels(2.1,3.2,np.deg2rad(6),np.deg2rad(.25), sps.ttest_ind(renc180,rschz180)[1])
sig_labels(0.8,3.2,np.deg2rad(7.5),np.deg2rad(.25), sps.ttest_ind(rctrl180,rschz180)[1])
plt.xlim([0,4])
plt.ylim(np.deg2rad([-8,8]))
plt.yticks(np.deg2rad([-8,0,8]),[-8,0,8])
plt.xticks([1,2,3],['ctrl','enc','schz'])
plt.plot([0,4],np.zeros(2),'k--', alpha=.3)
ax8.get_yaxis().set_tick_params(direction='in')
ax8.get_xaxis().set_tick_params(direction='in')
sns.despine()
plt.tight_layout(w_pad=1.5)


# Cohen's d
print(cohen_d(rctrl60, rschz60))
print(cohen_d(renc60, rschz60))

print(cohen_d(rctrl180, renc180))
print(cohen_d(rctrl180, rschz180))
print(cohen_d(renc180, rschz180))
