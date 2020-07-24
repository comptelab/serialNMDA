# FIGURE 2
# MODEL SCHEME

# author: heike stein
# last mod: 23/04/20

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import seaborn as sns


np.set_printoptions(precision=4)
sns.set_context("talk", font_scale=1)
sns.set_style("ticks")

uglybrown   = sns.xkcd_rgb["ugly brown"]
darkmauve   = sns.xkcd_rgb["dark mauve"]
darkorange  = sns.xkcd_rgb["dark orange"]

#########################################################################
#                           LOAD DATA                                   #
#########################################################################

i,t     = np.load('../data/Fig_2/FIGURE2_spikes.npy')
dec     = np.load('../data/Fig_2/FIGURE2_decoder.npy')
frs     = np.loadtxt('../data/Fig_2/FIGURE2_firing_rates.txt',dtype="str")[:1000,1:].astype('float')
stps    = np.loadtxt('../data/Fig_2/FIGURE2_stp.txt',dtype="str")[:1000,1:].astype('float')
sims    = np.load('../data/Fig_2/FIGURE2_scheme.npy', allow_pickle=True, encoding='latin1').item()

train1  = np.array(sims['spike1'])
train2  = np.array(sims['spike2'])
potent  = np.array(sims['stp'])

ipotent = []; tpotent = []; dpotent = []
for k in range(len(train1)):
    if np.abs(train1[k]-train2).any()<100: 
        for j in range(len(train2)):
            if np.abs(train1[k]-train2[j])<100:
                ipotent.append(np.argmax([train1[k], train2[j]]))
                tpotent.append(np.max([train1[k], train2[j]]))
                dpotent.append(train1[k] - train2[j])

time = np.arange(0,101)
STP = np.concatenate([np.flip(np.exp(-time/20.),0)[:-1],np.exp(-time/20.)])
time = np.arange(-100,101)

# exclude trials with bumps at target n-1 location
out     = frs[:,6250]>10 

fr      = np.nanmean(frs[~out],0)
stp     = np.nanmean(stps[~out],0)


#########################################################################
#                                PLOT                                   #
#########################################################################

fig = plt.figure(figsize=(8,10))
gs1 = gs.GridSpec(4,3, height_ratios=[2,2,.01,1])

ax1 = plt.subplot(gs1[0,:])
ax2 = plt.subplot(gs1[1,:])
ax4 = plt.subplot(gs1[3,0])
ax5 = plt.subplot(gs1[3,1:])

ax1.plot(t, i, 'k.', ms=2)
ax1.fill_between([1.5,1.75], [0,0], [1024,1024], color=uglybrown, alpha=.3)
ax1.fill_between([2.75,3.], [0,0], [1024,1024], color=darkmauve, alpha=.3)
ax1.fill_between([6.,6.25], [0,0], [1024,1024], color=uglybrown, alpha=.3)
ax1.plot(np.linspace(1.5,2.75,len(dec[15:28])),(dec/(2*np.pi)*1024)[15:28],color=darkorange)
ax1.plot(np.linspace(6,9.25,len(dec[60:])),(dec/(2*np.pi)*1024)[60:],color=darkorange)
ax1.plot([6,9.25],[(135/360.)*1024,(135/360.)*1024],'--',color=darkorange)
ax1.plot(1.5,(180/360.)*1024,'>', ms=15, color=darkorange)
ax1.plot(6,(135/360.)*1024,'>', ms=15, color=darkorange)
ax1.set_xlim([0,9.25])
ax1.set_ylim([0,1024])
ax1.set_ylabel('neuron label')
ax1.set_yticks([0,1024/2, 1024])
ax1.set_yticklabels([r'$-180^\circ$', r'$0^\circ$', r'$180^\circ$'])
ax1.set_xticks(np.arange(0,9.2,1))
ax1.set_xticklabels([])#np.arange(-6,4))
ax1.get_xaxis().set_tick_params(direction='in')
ax1.get_yaxis().set_tick_params(direction='in')

ax2.plot(np.linspace(125,9125,len(fr)),fr,'k', alpha=1)
ax2.plot(np.zeros(len(stp)),'k--', alpha=.3)
ax2.set_ylabel('firing rate (spikes/s)')
ax2.set_yticks([0,20,40,60])
ax2.fill_between([1500,1750], [-2,-2], [60,60], color=uglybrown, alpha=.3)
ax2.fill_between([2750,3000], [-2,-2], [60,60], color=darkmauve, alpha=.3)
ax2.fill_between([6000,6250], [-2,-2], [60,60], color=uglybrown, alpha=.3)
ax2.get_xaxis().set_tick_params(direction='in')
ax2.get_yaxis().set_tick_params(direction='in')
ax2.set_xlabel(r'time from current stimulus (s)')
ax2.text(1600,62,'trial n-1')
ax2.text(4400,62,'ITI')
ax2.text(6700,62,'trial n')
ax2.set_xticks(np.arange(0,9200,1000))
ax2.set_xticklabels(np.arange(-6,4))
ax2.set_ylim([-2,60])
sns.despine(ax=ax2)

ax3 = ax2.twinx()
ax3.plot(stp,color=darkorange)
ax3.set_ylabel(r'$w_{ij}$', color=darkorange)
ax3.tick_params(axis='y', labelcolor=darkorange)
ax3.tick_params('y', color=darkorange)
ax3.set_xlim([0,len(stp)])
plt.ylim(0.99965,1.01)
plt.yticks([1,1.005,1.01], ['1.00','','1.01'])
ax3.get_yaxis().set_tick_params(direction='in')
ax3.spines['right'].set_color(darkorange)
sns.despine(ax=ax3,right=False)

ax4.plot(time, STP*22.+4, 'k', lw=2, alpha=.8, label=r'$P=2.2$')#r'$P=2.2 \times 10^{-4}$')
ax4.plot(time, STP*12.+4, uglybrown, lw=2, alpha=.8, label=r'$P=1.2$')#label=r'$P=1.2 \times 10^{-4}$')
ax4.plot(time, STP*2.+4, darkmauve, lw=2, alpha=.8, label=r'$P=0.2$')#label=r'$P=0.2 \times 10^{-4}$')
ax4.text(50, .5*22, r'$ \times 10^{-4}$', fontsize=12)
ax4.plot(time, np.zeros(len(time))+4, 'k--', alpha=.2)
ax4.set_ylim([0,1.3*22])
ax4.set_xlim([-100,100])
ax4.plot([-100,100],[0,0], 'k-')
ax4.set_xlabel(r'$t_{j}-t_{i} (ms)$')
ax4.set_ylabel(r'$\Delta_{w}$', color=darkorange)
ax4.set_xticks([-100,0,100])
ax4.set_yticks([])
ax4.legend(fontsize=12, loc=1)
ax4.get_xaxis().set_tick_params(direction='in')
ax4.spines['left'].set_color(darkorange)
sns.despine(ax=ax4)

ax5.plot(-100,0.000375+np.max(potent),'ko', ms=24)
ax5.plot(-100,-0.000375+np.min(potent),'ko', ms=24)
ax5.plot(-100,np.min(potent),'k^', ms=16)
ax5.plot([-100,-100],[np.min(potent),0.000375+np.max(potent)], 'k-', lw=2)
for s in train1:
    ax5.plot([s,s],[0.00025, 0.0005]+np.max(potent),'k-')
ax5.text(-200,0.0003+np.max(potent), 'j')
ax5.text(-200,-0.00045+np.min(potent), 'i')
ax5.text(10,np.mean(potent)-0.0001, r'$w_{ij}$')
ax5.text(490,np.mean(potent)-0.0001, r'$\Delta_{w}$', fontsize=16, color=darkorange)
ax5.plot(range(1000), potent, 'k')
for s in train2:
    ax5.plot([s,s],[-0.0005, -0.00025]+np.min(potent),'k-')
ax5.plot([900,1000],[-0.0006+np.min(potent),-0.0006+np.min(potent)], 'k-', lw=4)
ax5.text(810,-0.00085+np.min(potent), '100 ms', fontsize=16)
ax5.set_xticks([])
ax5.set_yticks([])
ax5.set_xlim(-250,1005)
sns.despine(ax=ax5, bottom=True,left=True)
gs1.tight_layout(fig, h_pad=0.0, w_pad=0.45)
