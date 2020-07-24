# FIGURE 3
# MODELING RESULTS

# author: heike stein
# last mod: 23/04/20

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import helpers as hf
import pandas as pd
import scikits.bootstrap as boot

np.set_printoptions(precision=4)
sns.set_context("talk", font_scale=1)
sns.set_style("ticks")

uglybrown   = sns.xkcd_rgb["ugly brown"]
darkmauve   = sns.xkcd_rgb["dark mauve"]
darkorange  = sns.xkcd_rgb["dark orange"]

#########################################################################
#                           FUNCTIONS                                   #
#########################################################################

def plot_ci(ax,x,cis,capsize,color):
    for c,ci in enumerate(cis):
        ax.plot([x[c],x[c]],ci, '-', color=color, lw=2, alpha=.8)
        ax.plot([x[c]-capsize, x[c]+capsize], [ci[0],ci[0]], '-', \
            color=color, lw=2, alpha=.8)
        ax.plot([x[c]-capsize, x[c]+capsize], [ci[1],ci[1]], '-', \
            color=color, lw=2, alpha=.8)


#########################################################################
#                           LOAD DATA                                   #
#########################################################################

bias_ctrl = np.loadtxt('../data/Fig_3/FIGURE3_beh_Apre0.00022.txt', dtype='S')[:,1:].astype('float')  #1 #22
bias_enc  = np.loadtxt('../data/Fig_3/FIGURE3_beh_Apre0.00012.txt' ,dtype='S')[:,1:].astype('float')  #5 #12
bias_schz = np.loadtxt('../data/Fig_3/FIGURE3_beh_Apre0.00002.txt', dtype='S')[:,1:].astype('float')  #11 #02

apre_bias = np.load('../data/Fig_3/FIGURE3_bias_Apre.npy')
apre_bias_ci = np.load('../data/Fig_3/FIGURE3_bias_ci_Apre.npy')
apre_prec = np.load('../data/Fig_3/FIGURE3_prec_Apre.npy')
apre_xticks = np.load('../data/Fig_3/FIGURE3_xticks_Apre.npy')
Apre_outliers = np.load('../data/Fig_3/FIGURE3_outliers_Apre.npy')

gee_bias = np.load('../data/Fig_3/FIGURE3_bias_gee.npy')
gee_bias_ci = np.load('../data/Fig_3/FIGURE3_bias_ci_gee.npy')
gee_prec = np.load('../data/Fig_3/FIGURE3_prec_gee.npy')
gee_xticks = np.load('../data/Fig_3/FIGURE3_xticks_gee.npy')
gee_outliers = np.load('../data/Fig_3/FIGURE3_outliers_gee.npy')

gei_bias = np.load('../data/Fig_3/FIGURE3_bias_gei.npy')
gei_bias_ci = np.load('../data/Fig_3/FIGURE3_bias_ci_gei.npy')
gei_prec = np.load('../data/Fig_3/FIGURE3_prec_gei.npy')
gei_xticks = np.load('../data/Fig_3/FIGURE3_xticks_gei.npy')
gei_outliers = np.load('../data/Fig_3/FIGURE3_outliers_gei.npy')

arand   = np.load('../data/Fig_3/FIGURE1_coef_rand.npy')
dat     = pd.read_pickle('../data/Fig_3/FIGURE1_raw_dat.pkl')

# group identifiers
ctrl    = np.array([s[0] for s in list(dat.subject.unique())]) == 'C'
enc     = np.array([s[0] for s in list(dat.subject.unique())]) == 'E'
schz    = np.array([s[0] for s in list(dat.subject.unique())]) == 'S'


#########################################################################
#                         GET RANDOM EFFECTS                            #
#########################################################################

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


#########################################################################
#                          PLOT SERIAL BIAS                             #
#########################################################################

window      = np.pi/3
step        = np.pi/30
xxx         = np.arange(0, np.pi, step)

sb_ctrl = np.zeros([3,len(xxx)]); se_ctrl = np.zeros([3,len(xxx)])
sb_enc  = np.zeros([3,len(xxx)]); se_enc  = np.zeros([3,len(xxx)])
sb_schz = np.zeros([3,len(xxx)]); se_schz = np.zeros([3,len(xxx)])

for bi, beh in enumerate([bias_ctrl,bias_enc,bias_schz]):
    for i in range(3):
        dists   = hf.circdist(beh[:,0], beh[:,1])
        err     = hf.circdist(beh[:,2+i], beh[:,1]) #resp, target
        out     = ((err>1) | (err<-1))
        if bi == 0: sb_ctrl[i,:], se_ctrl[i,:] = \
                    hf.folded_bias(dists[~out], err[~out], window, step)
        if bi == 1: sb_enc[i,:], se_enc[i,:] = \
                    hf.folded_bias(dists[~out], err[~out], window, step)
        if bi == 2: sb_schz[i,:], se_schz[i,:] = \
                    hf.folded_bias(dists[~out], err[~out], window, step)

#########################################################################
#                          PLOT SERIAL BIAS                             #
#########################################################################

fig, ((ax1, ax3, ax5), (ax7, ax8, ax9), (ax10, ax11, ax12)) = \
    plt.subplots(3,3, figsize=(12,12))

ax1.plot(np.arange(len(apre_bias[:,2])), apre_bias[:,2], color=darkorange)
ax1.plot(np.arange(len(apre_bias[:,2])), np.zeros(len(apre_bias[:,2])), 'k--', alpha=.3)
ax1.fill_between(np.arange(len(apre_bias_ci[:,2])), np.array(apre_bias_ci[:,2,0]), 
    np.array(apre_bias_ci[:,2,1]), color=darkorange, alpha=.3)
ax1.set_xlabel(r'potentiation factor $P$')
ax1.set_ylabel(r'bias strength ($^\circ$)', color=darkorange)
ax1.tick_params(axis='y', labelcolor=darkorange)
ax1.tick_params('y', colors=darkorange)
ax1.set_ylim(-10,10)
ax1.set_yticks([-10,-5,0,5,10])
ax1.get_yaxis().set_tick_params(direction='in')
ax1.get_xaxis().set_tick_params(direction='in')
sns.despine(ax=ax1)

ax2 = ax1.twinx()
ax2.plot(np.arange(len(apre_prec[:,0])), apre_prec[:,0], color='k')
ax2.fill_between(np.arange(len(apre_prec[:,0])), np.array(apre_prec[:,1:]).T[0], 
    np.array(apre_prec[:,1:]).T[1], color='k', alpha=.3)
ax2.plot([1],[2],'^', ms=15, color=darkmauve)
ax2.plot([5],[2],'^', ms=15, color=uglybrown)
ax2.plot([11],[2],'^', ms=15, color='k')
ax2.tick_params('y', colors='k', labelcolor='k')
ax2.set_xticks(np.arange(len(apre_prec[:,0]))[::4])
ax2.set_xticklabels(apre_xticks.astype("int"))
ax2.set_yticklabels([])
ax2.get_xaxis().set_tick_params(direction='in')
ax2.get_yaxis().set_tick_params(direction='in')
ax2.set_xlim([0,len(apre_prec[:,0])-1])
ax2.set_ylim(0,40)
sns.despine(ax=ax2,right=False)

ax3.plot(np.arange(len(gei_bias[:,2])), gei_bias[:,2], color=darkorange)
ax3.plot(np.arange(len(gei_bias[:,2])), np.zeros(len(gei_bias[:,2])), 'k--', alpha=.3)
ax3.text(.5,-8,r'unstable', fontsize=14)
ax3.text(len(gei_bias[:,2])-7,-8,r'unstable', fontsize=14)
ax3.plot([np.where(np.array(gei_outliers)<10)[0][0]-.5,np.where(np.array(gei_outliers)<10)[0][0]-.5],[-10,10], 'k--')
ax3.plot([np.where(np.array(gei_outliers)<10)[0][-1]+.5,np.where(np.array(gei_outliers)<10)[0][-1]+.5],[-10,10], 'k--')
ax3.fill_between(np.arange(len(gei_bias_ci[:,2,0])), np.array(gei_bias_ci[:,2,0]), 
    np.array(gei_bias_ci[:,2,1]), color=darkorange, alpha=.3)
ax3.set_xlabel(r'$g_{EI}$ ($\%$ change)')
ax3.tick_params(axis='y', labelcolor=darkorange)
ax3.tick_params('y', colors=darkorange)
ax3.set_ylim(-10,10)
ax3.set_yticklabels([])
ax3.set_yticks([-10,-5,0,5,10])
ax3.get_yaxis().set_tick_params(direction='in')
ax3.get_xaxis().set_tick_params(direction='in')
sns.despine(ax=ax3)

ax4 = ax3.twinx()
ax4.plot(np.arange(len(gei_prec[:,0])), gei_prec[:,0], color='k')
ax4.fill_between(np.arange(len(gei_prec[:,0])), np.array(gei_prec[:,1:]).T[0], 
    np.array(gei_prec[:,1:]).T[1], color='k', alpha=.3)
ax4.set_yticklabels([])
ax4.plot([6],[2],'^', ms=15, color=darkmauve)
ax4.plot([8],[2],'^', ms=15, color=uglybrown)
ax4.plot([10],[2],'^', ms=15, color='k')
ax4.set_xticks(np.arange(len(gei_prec[:,0]))[::4])
ax4.set_xticklabels(gei_xticks)
ax4.get_xaxis().set_tick_params(direction='in')
ax4.get_yaxis().set_tick_params(direction='in')
ax4.set_xlim([0,len(gei_prec[:,0])-1])
ax4.set_ylim(0,40)
sns.despine(ax=ax4,right=False)


ax5.plot(np.arange(len(gee_bias[:,2])), gee_bias[:,2], color=darkorange)
ax5.plot(np.arange(len(gee_bias[:,2])), np.zeros(len(gee_bias[:,2])), 'k--', alpha=.3)
ax5.text(.5,-8,r'unstable', fontsize=14)
ax5.text(len(gee_bias[:,2])-7,-8,r'unstable', fontsize=14)
ax5.plot([np.where(np.array(gee_outliers)<10)[0][0]-.5,np.where(np.array(gee_outliers)<10)[0][0]-.5],[-10,10], 'k--')
ax5.plot([np.where(np.array(gee_outliers)<10)[0][-1]+.5,np.where(np.array(gee_outliers)<10)[0][-1]+.5],[-10,10], 'k--')
ax5.fill_between(np.arange(len(gee_bias_ci[:,2])), np.array(gee_bias_ci[:,2,0]), 
    np.array(gee_bias_ci[:,2,1]), color=darkorange, alpha=.3)
ax5.set_xlabel(r'$g_{EE}$ ($\%$ change)')
ax5.set_yticklabels([])
ax5.tick_params(axis='y', labelcolor=darkorange)
ax5.tick_params('y', colors=darkorange)
ax5.set_ylim(-10,10)
ax5.set_yticks([-10,-5,0,5,10],[])
ax5.get_yaxis().set_tick_params(direction='in')
ax5.get_xaxis().set_tick_params(direction='in')
sns.despine(ax=ax5)

ax6 = ax5.twinx()
ax6.plot(np.arange(len(gee_prec[:,0])), gee_prec[:,0], color='k')
ax6.fill_between(np.arange(len(gee_prec[:,0])), np.array(gee_prec[:,1:]).T[0], 
    np.array(gee_prec[:,1:]).T[1], color='k', alpha=.3)
ax6.plot([7],[2],'^', ms=15, color=darkmauve)
ax6.plot([8],[2],'^', ms=15, color=uglybrown)
ax6.plot([10],[2],'^', ms=15, color='k')
ax6.set_ylabel(r'circular s.d. of errors $(^\circ)$', color='k')
ax6.set_xticks(np.arange(len(gee_prec[:,0]))[::4])
ax6.set_xticklabels(gee_xticks)
ax6.get_xaxis().set_tick_params(direction='in')
ax6.get_yaxis().set_tick_params(direction='in')
ax6.set_xlim([0,len(gee_prec[:,0])-1])
ax6.set_ylim(0,40)
sns.despine(ax=ax6,right=False)

ax7.plot([9,10,11],apre_bias[11],'o-', color='k', alpha=.8)
ax7.plot([5,6,7],apre_bias[5],'o-', color=uglybrown, alpha=.8)
ax7.plot([1,2,3],apre_bias[1],'o-', color=darkmauve, alpha=.8)
ax7.plot([0,12],[0,0],'k--', alpha=.3)
plot_ci(ax7, [9,10,11], np.degrees([boot.ci(rctrl0),boot.ci(rctrl60),boot.ci(rctrl180)]), capsize=.3, color='k')
plot_ci(ax7, [5,6,7], np.degrees([boot.ci(renc0),boot.ci(renc60),boot.ci(renc180)]), capsize=.3,color=uglybrown)
plot_ci(ax7, [1,2,3], np.degrees([boot.ci(rschz0),boot.ci(rschz60),boot.ci(rschz180)]), capsize=.3, color=darkmauve)
ax7.set_xticks([2,6,10])
ax7.set_xticklabels(['schz','enc','ctrl'])
ax7.set_ylim([-8,8])
ax7.set_yticks([-8,0,8])
ax7.set_title(r'$P$')
ax7.text(.75,5, 'delay', fontsize=16)
ax7.text(.25,4, '0s', fontsize=12)
ax7.text(1.5,4, '1s', fontsize=12)
ax7.text(2.75,4, '3s', fontsize=12)
ax7.set_ylabel(r'bias strength ($^\circ$)')
ax7.get_xaxis().set_tick_params(direction='in')
ax7.get_yaxis().set_tick_params(direction='in')
sns.despine(ax=ax7)

ax8.plot([9,10,11],gei_bias[10],'o-', color='k', alpha=.8)
ax8.plot([5,6,7],gei_bias[8],'o-', color=uglybrown, alpha=.8)
ax8.plot([1,2,3],gei_bias[6],'o-', color=darkmauve, alpha=.8)
ax8.plot([0,12],[0,0],'k--', alpha=.3)
plot_ci(ax8, [9,10,11], np.degrees([boot.ci(rctrl0),boot.ci(rctrl60),boot.ci(rctrl180)]), capsize=.3, color='k')
plot_ci(ax8, [5,6,7], np.degrees([boot.ci(renc0),boot.ci(renc60),boot.ci(renc180)]), capsize=.3,color=uglybrown)
plot_ci(ax8, [1,2,3], np.degrees([boot.ci(rschz0),boot.ci(rschz60),boot.ci(rschz180)]), capsize=.3, color=darkmauve)
ax8.set_xticks([2,6,10])
ax8.set_xticklabels(['schz','enc','ctrl'])
ax8.set_ylim([-8,8])
ax8.set_yticks([-8,0,8])
ax8.set_title(r'$g_{EI}$')
ax8.get_xaxis().set_tick_params(direction='in')
ax8.get_yaxis().set_tick_params(direction='in')
sns.despine(ax=ax8)

ax9.plot([9,10,11],gee_bias[10],'o-', color='k', alpha=.8)
ax9.plot([5,6,7],gee_bias[8],'o-', color=uglybrown, alpha=.8)
ax9.plot([1,2,3],gee_bias[7],'o-', color=darkmauve, alpha=.8)
ax9.plot([0,12],[0,0],'k--', alpha=.3)
plot_ci(ax9, [9,10,11], np.degrees([boot.ci(rctrl0),boot.ci(rctrl60),boot.ci(rctrl180)]), capsize=.3, color='k')
plot_ci(ax9, [5,6,7], np.degrees([boot.ci(renc0),boot.ci(renc60),boot.ci(renc180)]), capsize=.3,color=uglybrown)
plot_ci(ax9, [1,2,3], np.degrees([boot.ci(rschz0),boot.ci(rschz60),boot.ci(rschz180)]), capsize=.3, color=darkmauve)
ax9.set_xticks([2,6,10])
ax9.set_xticklabels(['schz','enc','ctrl'])
ax9.set_ylim([-8,8])
ax9.set_yticks([-8,0,8])
ax9.set_title(r'$g_{EE}$')
ax9.get_xaxis().set_tick_params(direction='in')
ax9.get_yaxis().set_tick_params(direction='in')
sns.despine(ax=ax9)

ax10.plot(xxx,sb_ctrl[0], '-', color='k')
ax10.plot(xxx,sb_enc[0], '-', color=uglybrown)
ax10.plot(xxx,sb_schz[0], '-', color=darkmauve)
ax10.fill_between(xxx,sb_ctrl[0]+se_ctrl[0], sb_ctrl[0]-se_ctrl[0], color='k', alpha=.3, label=r'$P=2.2 \times 10^{-4}$')
ax10.fill_between(xxx,sb_enc[0]+se_enc[0], sb_enc[0]-se_enc[0], color=uglybrown, alpha=.3, label=r'$P=1.2 \times 10^{-4}$')
ax10.fill_between(xxx,sb_schz[0]+se_schz[0], sb_schz[0]-se_schz[0], color=darkmauve, alpha=.3, label=r'$P=0.2 \times 10^{-4}$')
ax10.plot(xxx,np.zeros(len(xxx)), 'k--', alpha=.3)
ax10.set_ylim([-.04,.04])
ax10.set_xlim([0,np.pi])
ax10.set_xticks([0,np.pi/2,np.pi])
ax10.set_xticklabels([0,90,180])
ax10.set_yticks([-np.deg2rad(3),0,np.deg2rad(3)])
ax10.set_yticklabels([-3,0,3])
ax10.set_ylabel(r'error in current trial $\theta^{e}$ ($^\circ$)')
ax10.set_title('0 seconds delay')
ax10.get_yaxis().set_tick_params(direction='in')
ax10.get_xaxis().set_tick_params(direction='in')
# ax10.legend()
sns.despine(ax=ax10)

ax11.plot(xxx,sb_ctrl[1], '-', color='k')
ax11.plot(xxx,sb_enc[1], '-', color=uglybrown)
ax11.plot(xxx,sb_schz[1], '-', color=darkmauve)
ax11.fill_between(xxx,sb_ctrl[1]+se_ctrl[1], sb_ctrl[1]-se_ctrl[1], color='k', alpha=.3)
ax11.fill_between(xxx,sb_enc[1]+se_enc[1], sb_enc[1]-se_enc[1], color=uglybrown, alpha=.3)
ax11.fill_between(xxx,sb_schz[1]+se_schz[1], sb_schz[1]-se_schz[1], color=darkmauve, alpha=.3)
ax11.plot(xxx,np.zeros(len(xxx)), 'k--', alpha=.3)
ax11.set_ylim([-.04,.04])
ax11.set_xlim([-np.pi,np.pi])
ax11.set_xlim([0,np.pi])
ax11.set_xticks([0,np.pi/2,np.pi])
ax11.set_xticklabels([0,90,180])
ax11.set_yticks([-np.deg2rad(3),0,np.deg2rad(3)])
ax11.set_yticklabels([-3,0,3])
ax11.set_xlabel(r'absolute distance $|\theta^{d}|$ ($^\circ$)')
ax11.set_title('1 second delay')
ax11.get_yaxis().set_tick_params(direction='in')
ax11.get_xaxis().set_tick_params(direction='in')
sns.despine(ax=ax11)

ax12.plot(xxx,sb_ctrl[2], '-', color='k')
ax12.plot(xxx,sb_enc[2], '-', color=uglybrown)
ax12.plot(xxx,sb_schz[2], '-', color=darkmauve)
ax12.fill_between(xxx,sb_ctrl[2]+se_ctrl[2], sb_ctrl[2]-se_ctrl[2], color='k', alpha=.3)
ax12.fill_between(xxx,sb_enc[2]+se_enc[2], sb_enc[2]-se_enc[2], color=uglybrown, alpha=.3)
ax12.fill_between(xxx,sb_schz[2]+se_schz[2], sb_schz[2]-se_schz[2], color=darkmauve, alpha=.3)
ax12.plot(xxx,np.zeros(len(xxx)), 'k--', alpha=.3)
ax12.set_ylim([-.04,.04])
ax12.set_xlim([-np.pi,np.pi])
ax12.set_xlim([0,np.pi])
ax12.set_xticks([0,np.pi/2,np.pi])
ax12.set_xticklabels([0,90,180])
ax12.set_yticks([-np.deg2rad(3),0,np.deg2rad(3)])
ax12.set_yticklabels([-3,0,3])
ax12.set_title('3 seconds delay')
ax12.get_yaxis().set_tick_params(direction='in')
ax12.get_xaxis().set_tick_params(direction='in')
sns.despine(ax=ax12)
plt.tight_layout()
