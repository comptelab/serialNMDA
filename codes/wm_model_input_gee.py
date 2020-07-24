"""
Implementation of a working memory model.
Literature:
Compte, A., Brunel, N., Goldman-Rakic, P. S., & Wang, X. J. (2000). Synaptic mechanisms and
network dynamics underlying spatial working memory in a cortical network model.
Cerebral Cortex, 10(9), 910-923.

Some parts of this implementation are inspired by material from
*Stanford University, BIOE 332: Large-Scale Neural Modeling, Kwabena Boahen & Tatiana Engel, 2013*,
online available.

Note: Most parameters differ from the original publication.
"""

# This file is part of the exercise code repository accompanying
# the book: Neuronal Dynamics (see http://neuronaldynamics.epfl.ch)
# located at http://github.com/EPFL-LCN/neuronaldynamics-exercises.

# This free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License 2.0 as published by the
# Free Software Foundation. You should have received a copy of the
# GNU General Public License along with the repository. If not,
# see http://www.gnu.org/licenses/.

# Should you reuse and publish the code for your own purposes,
# please cite the book or point to the webpage http://neuronaldynamics.epfl.ch.

# Wulfram Gerstner, Werner M. Kistler, Richard Naud, and Liam Paninski.
# Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition.
# Cambridge University Press, 2014.

from brian2 import ms, Hz, namp, nF, nS, mV, kHz
from brian2 import NeuronGroup, Synapses, PoissonInput, network_operation, defaultclock, run, prefs, devices
from brian2.monitors import StateMonitor, SpikeMonitor, PopulationRateMonitor
from random import sample
from collections import deque
import numpy as np
import math
import cmath
from scipy.special import erf
from numpy.fft import rfft, irfft
import os
import sys
import time
import socket
import random
from joblib import Parallel, delayed
import multiprocessing as mp


prefs.codegen.target = 'cython'

start_time = time.time()
numcores   = mp.cpu_count()

par = int(sys.argv[1])
pars = np.arange(0.98,1.021,.002)
par = pars[par]

repstrength = 1.25
  
defaultclock.dt = 0.05 * ms



##########################################################################################33
#                       DEFINE SIMULATIONS
##########################################################################################33


def simulate_wm(
        N_excitatory=2048, N_inhibitory=512,
        N_extern_poisson=1000, poisson_firing_rate=1.8 * Hz, 
        sigma_weight_profile=14.4, Jpos_excit2excit=1.63,
        stimulus1_center_deg=180, stimulus2_center_deg=235, 
        stimulus_width_deg=60, stimulus_strength=0.07 * namp,
        t_stimulus1_start=0 * ms, t_stimulus2_start=4000 * ms,
        t_stimulus_duration=0 * ms,
        t_delay1=3000 * ms, t_delay2=3000 * ms, 
        t_iti_duration=300 * ms, sim_time=2000. * ms,
        monitored_subset_size=1024):
    """
    Args:
        N_excitatory (int): Size of the excitatory population
        N_inhibitory (int): Size of the inhibitory population
        weight_scaling_factor (float): weight prefactor. When increasing the size of the populations,
            the synaptic weights have to be decreased. Using the default values, we have
            N_excitatory*weight_scaling_factor = 2048 and N_inhibitory*weight_scaling_factor=512
        N_extern_poisson (int): Size of the external input population (Poisson input)
        poisson_firing_rate (Quantity): Firing rate of the external population
        sigma_weight_profile (float): standard deviation of the gaussian input profile in
            the excitatory population.
        Jpos_excit2excit (float): Strength of the recurrent input within the excitatory population.
            Jneg_excit2excit is computed from sigma_weight_profile, Jpos_excit2excit and the normalization
            condition.
        stimulus_center_deg (float): Center of the stimulus in [0, 360]
        stimulus_width_deg (float): width of the stimulus. All neurons in
            stimulus_center_deg +- (stimulus_width_deg/2) receive the same input current
        stimulus_strength (Quantity): Input current to the neurons at stimulus_center_deg +- (stimulus_width_deg/2)
        t_stimulus_start (Quantity): time when the input stimulus is turned on
        t_stimulus_duration (Quantity): duration of the stimulus.
        monitored_subset_size (int): nr of neurons for which a Spike- and Voltage monitor is registered.
        sim_time (Quantity): simulation time

    Returns:

       results (tuple):
       rate_monitor_excit (Brian2 PopulationRateMonitor for the excitatory population),
        spike_monitor_excit, voltage_monitor_excit, idx_monitored_neurons_excit,\
        rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib,\
        weight_profile_45 (The weights profile for the neuron with preferred direction = 45deg).
    """

    global par

    print par

    devices.device.seed(os.getpid())
    # specify the excitatory pyramidal cells:
    Cm_excit                    = 0.5 * nF  # membrane capacitance of excitatory neurons
    G_leak_excit                = 25.0 * nS  # leak conductance
    E_leak_excit                = -70.0 * mV  # reversal potential
    v_firing_threshold_excit    = -50.0 * mV  # spike condition
    v_reset_excit               = -60.0 * mV  # reset voltage after spike
    t_abs_refract_excit         = 2.0 * ms  # absolute refractory period

    # specify the weight profile in the recurrent population
    # std-dev of the gaussian weight profile around the prefered direction
    # sigma_weight_profile = 12.0  # std-dev of the gaussian weight profile around the prefered direction
    # Jneg_excit2excit = 0

    # specify the inhibitory interneurons:
    Cm_inhib                    = 0.2 * nF
    G_leak_inhib                = 20.0 * nS
    E_leak_inhib                = -70.0 * mV
    v_firing_threshold_inhib    = -50.0 * mV
    v_reset_inhib               = -60.0 * mV
    t_abs_refract_inhib         = 1.0 * ms

    # specify the AMPA synapses
    E_AMPA      = 0.0 * mV
    tau_AMPA    = 2.0 * ms

    # specify the GABA synapses
    E_GABA      = -70.0 * mV
    tau_GABA    = 10.0 * ms

    # specify the NMDA synapses
    E_NMDA      = 0.0 * mV
    tau_NMDA_s  = 100.0 * ms 
    tau_NMDA_x  = 2.0 * ms
    alpha_NMDA  = 0.5 * kHz

    weight_scaling_factor = 2048./N_excitatory

    # projections from the external population
    G_extern2inhib  = 2.38 * nS
    G_extern2excit  = 3.1 * nS

    # projectsions from the inhibitory populations
    G_inhib2inhib   = weight_scaling_factor * 1.024 * nS
    G_inhib2excit   = weight_scaling_factor * 1.336 * nS

    # projections from the excitatory population NMDA
    G_excit2excit   = par*weight_scaling_factor * 0.28 * nS #nmda+ampa
    G_excit2inhib   = weight_scaling_factor * 0.212 * nS  # nmda+ampa

    # recurrent AMPA
    G_excit2excitA  = weight_scaling_factor * 1.* 0.251 * nS  #ampa
    GEEA            = G_excit2excitA/G_extern2excit
    G_excit2inhibA  = weight_scaling_factor * 0.192 * nS  #ampa
    GEIA            = G_excit2inhibA/G_extern2inhib

    # STDP
    taupre      = 20 * ms
    taupost     = 20 * ms
    wmax        = 2. #1.1
    Apre        = 0.00022 #0.00025 #set to zero to deactivate STDP # 0.02 
    Apost       = Apre #-Apre *taupre/taupost*1.03 #1.4 #negative for LTD, positive for LTP
    stp_decay   = 0.04#25 #0.025
#    Apost      = Apre *taupre/taupost #negative for LTD, positive for LTP

    # compute the simulus index
    stim1_center_idx = int(round(N_excitatory / 360. * stimulus1_center_deg))
    stim1_width_idx  = int(round(N_excitatory / 360. * stimulus_width_deg / 2))
    stim1_target_idx = [idx % N_excitatory
                       for idx in
                       range(stim1_center_idx - stim1_width_idx, stim1_center_idx + stim1_width_idx + 1)]

    stim2_center_idx = int(round(N_excitatory / 360. * stimulus2_center_deg))
    stim2_width_idx  = int(round(N_excitatory / 360. * stimulus_width_deg / 2))
    stim2_target_idx = [idx % N_excitatory
                       for idx in
                       range(stim2_center_idx - stim2_width_idx, stim2_center_idx + stim2_width_idx + 1)]

    # precompute the weight profile for the recurrent population
    tmp                     = math.sqrt(2. * math.pi) * sigma_weight_profile * erf(180. / math.sqrt(2.) / sigma_weight_profile) / 360.
    Jneg_excit2excit        = (1. - Jpos_excit2excit * tmp) / (1. - tmp)
    presyn_weight_kernel    = [(Jneg_excit2excit + (Jpos_excit2excit - Jneg_excit2excit) *
                               math.exp(-.5 * (360. * min(nj, N_excitatory - nj) / N_excitatory) ** 2 / sigma_weight_profile ** 2))
                               for nj in 
                               range(N_excitatory)]
    fft_presyn_weight_kernel = rfft(presyn_weight_kernel)

    # define the inhibitory population
    a = 0.062/mV
    inhib_lif_dynamics = """
        s_NMDA_total : 1  # the post synaptic sum of s. compare with s_NMDA_presyn
        dv/dt = (
        - G_leak_inhib * (v-E_leak_inhib)
        - G_extern2inhib * s_AMPA * (v-E_AMPA)
        - G_inhib2inhib * s_GABA * (v-E_GABA)
        - G_excit2inhib * s_NMDA_total * (v-E_NMDA)/(1.0+1.0*exp(-a*v)/3.57)
        )/Cm_inhib : volt (unless refractory)
        ds_AMPA/dt = -s_AMPA/tau_AMPA : 1
        ds_GABA/dt = -s_GABA/tau_GABA : 1
    """

    inhib_pop = NeuronGroup(
        N_inhibitory, model=inhib_lif_dynamics,
        threshold="v>v_firing_threshold_inhib", reset="v=v_reset_inhib", refractory=t_abs_refract_inhib,
        method="rk2")
    # initialize with random voltages:
    inhib_pop.v     = np.random.uniform(v_reset_inhib / mV, high=v_firing_threshold_inhib / mV,
                                       size=N_inhibitory) * mV
    # set the connections: inhib2inhib
    syn_inhib2inhib = Synapses(inhib_pop, target=inhib_pop, on_pre="s_GABA += 1.0", delay=0.0 * ms)
    syn_inhib2inhib.connect(condition="i!=j", p=1.0)
#    syn_inhib2inhib.connect(p=1.0)
    # set the connections: extern2inhib
    input_ext2inhib = PoissonInput(target=inhib_pop, target_var="s_AMPA",
                                   N=N_extern_poisson, rate=poisson_firing_rate, weight=1.0)

    # specify the excitatory population:
    excit_lif_dynamics = """
        I_stim : amp
        s_NMDA_total : 1  # the post synaptic sum of s. compare with s_NMDA_presyn
        dv/dt = (
        - G_leak_excit * (v-E_leak_excit)
        - G_extern2excit * s_AMPA * (v-E_AMPA)
        - G_inhib2excit * s_GABA * (v-E_GABA)
        - G_excit2excit * s_NMDA_total * (v-E_NMDA)/(1.0+1.0*exp(-a*v)/3.57)
        + I_stim
        )/Cm_excit : volt (unless refractory)
        ds_AMPA/dt = -s_AMPA/tau_AMPA : 1
        ds_GABA/dt = -s_GABA/tau_GABA : 1
        ds_NMDA/dt = -s_NMDA/tau_NMDA_s + alpha_NMDA * x * (1-s_NMDA) : 1
        dx/dt = -x/tau_NMDA_x : 1
    """

    excit_pop = NeuronGroup(N_excitatory, model=excit_lif_dynamics,
                            threshold="v>v_firing_threshold_excit", reset="v=v_reset_excit",
                            refractory=t_abs_refract_excit, method="rk2")
    # initialize with random voltages:
    excit_pop.v = np.random.uniform(v_reset_excit / mV, high=v_firing_threshold_excit / mV,
                                       size=N_excitatory) * mV
    excit_pop.I_stim = 0. * namp
    # set the connections: extern2excit
    input_ext2excit = PoissonInput(target=excit_pop, target_var="s_AMPA",
                                   N=N_extern_poisson, rate=poisson_firing_rate, weight=1.0)

    # set the connections: inhibitory to excitatory
    syn_inhib2excit = Synapses(inhib_pop, excit_pop, on_pre="s_GABA += 1.0")
    syn_inhib2excit.connect(p=1.0)

    # set the connections: excitatory to inhibitory NMDA connections
    syn_excit2inhib = Synapses(excit_pop, inhib_pop,
                               model="s_NMDA_total_post = s_NMDA_pre : 1 (summed)", method="rk2")
    syn_excit2inhib.connect(p=1.0)

    # # set the connections: UNSTRUCTURED excitatory to excitatory
    # syn_excit2excit = Synapses(excit_pop, excit_pop,
    #        model= "s_NMDA_total_post = s_NMDA_pre : 1 (summed)", method="rk2")
    # syn_excit2excit.connect(condition="i!=j", p=1.)

    # set the STRUCTURED recurrent AMPA input
    # equations for weights, trace decay
    synapse_eqs = '''
    w : 1
    stp : 1
    dapre/dt = -apre/ taupre : 1 (event-driven)
    dapost/dt = -apost / taupost : 1 (event-driven)
    '''

    # equations for presynaptic spike
    eqs_pre = '''
    s_AMPA_post += w*stp
    x_pre += (1.0/N_excitatory)*stp
    apre += Apre
    stp = clip(stp + apost - stp_decay * (stp - 1.), 0, wmax)
    '''

    # equations for postsynaptic spike
    eqs_post = '''
    apost += Apost
    stp = clip(stp + apre, 0, wmax)
    '''

    syn_excit2excit = Synapses(excit_pop, excit_pop, synapse_eqs, on_pre=eqs_pre, on_post=eqs_post)
    syn_excit2excit.connect(condition='i!=j',p=1.0)
    syn_excit2excit.stp=1.0
    syn_excit2excit.w['abs(i-j)<N_excitatory/2'] = 'GEEA *(Jneg_excit2excit + (Jpos_excit2excit - Jneg_excit2excit) * exp(-.5 * (360. * abs(i-j) / N_excitatory) ** 2 / sigma_weight_profile ** 2))'
    syn_excit2excit.w['abs(i-j)>=N_excitatory/2'] = 'GEEA *(Jneg_excit2excit + (Jpos_excit2excit - Jneg_excit2excit) * exp(-.5 * (360. * (N_excitatory - abs(i-j)) / N_excitatory) ** 2 / sigma_weight_profile ** 2))'

    syn_excit2inhibA = Synapses(excit_pop, inhib_pop, model="w : 1", on_pre="s_AMPA_post += w")
    syn_excit2inhibA.connect(p=1.0)
    syn_excit2inhibA.w = GEIA


    # set the STRUCTURED recurrent NMDA input. use a network_operation
    @network_operation()
    def update_nmda_sum():
        fft_s_NMDA = rfft(excit_pop.s_NMDA)
        fft_s_NMDA_total = np.multiply(fft_presyn_weight_kernel, fft_s_NMDA)
        s_NMDA_tot = irfft(fft_s_NMDA_total,N_excitatory)
        excit_pop.s_NMDA_total_ = s_NMDA_tot
        # excit_pop.s_NMDA_total = s_NMDA_tot
        # inhib_pop.s_NMDA_total = fft_s_NMDA[0]

    @network_operation(dt=100 * ms)
    def time_counter(t):
        print(t)

    @network_operation(dt=1 * ms)
    def stimulate_network(t):
        if t >= t_stimulus1_start and t < t_stimulus1_start+t_stimulus_duration:
            excit_pop.I_stim[stim1_target_idx] = stimulus_strength
        elif t >= t_stimulus1_start+t_stimulus_duration and t < t_stimulus1_start+t_stimulus_duration+t_delay1:
            excit_pop.I_stim = 0. * namp
        elif t >= t_stimulus1_start+t_stimulus_duration+t_delay1 and t < t_stimulus1_start+t_stimulus_duration+t_delay1+t_stimulus_duration:
            excit_pop.I_stim = -1.*stimulus_strength
        elif t >= t_stimulus2_start-t_iti_duration and t < t_stimulus2_start:
            excit_pop.I_stim = 0. * namp
        elif t >= t_stimulus2_start and t < t_stimulus2_start+t_stimulus_duration:
            excit_pop.I_stim[stim2_target_idx] = stimulus_strength
        else:
            #syn_excit2excit.sgn=-1.0  # neuromodulation change
            excit_pop.I_stim = 0. * namp

    def get_monitors(pop, nr_monitored, N):
        nr_monitored = min(nr_monitored, (N))
        idx_monitored_neurons = [int(math.ceil(k))
             for k in np.linspace(0, N - 1, nr_monitored + 2)][1:-1]  # sample(range(N), nr_monitored)
        # rate_monitor    = PopulationRateMonitor(pop)
        spike_monitor   = SpikeMonitor(pop, record=idx_monitored_neurons)
        # voltage_monitor = StateMonitor(pop, "v", record=idx_monitored_neurons)
        synapse_monitor = StateMonitor(syn_excit2excit, "stp", record=syn_excit2excit[stim1_center_idx,stim1_center_idx-10:stim1_center_idx+10], dt=1*ms)
        # return rate_monitor, spike_monitor, voltage_monitor, idx_monitored_neurons, synapse_monitor
        return spike_monitor, synapse_monitor

    # collect data of a subset of neurons:
    spike_monitor_excit, synapse_monitor_excit = get_monitors(excit_pop, monitored_subset_size, N_excitatory)
    run(sim_time)

    return spike_monitor_excit, synapse_monitor_excit
    #return rate_monitor_excit, spike_monitor_excit, voltage_monitor_excit, idx_monitored_neurons_excit, synapse_monitor_excit, rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib, synapse_monitor_inhib, weight_profile_45


def decode(firing_rate, N_e):
    angles = np.arange(0,N_e)*2*np.pi/N_e
    R = []
    R = np.sum(np.dot(firing_rate,np.exp(1j*angles)))/N_e
    angle = np.angle(R)
    if angle < 0:
        angle +=2*np.pi 
    return angle 


def readout(i, t, sim_time, N_e):
    w1      = 100*ms
    w2      = 250*ms
    n_wins  = int((sim_time-w2)/w1)

    decs = []
    for ti in range(int(n_wins)):
        fr  = np.zeros(N_e)
        idx = ((t>ti*w1-w2/2) & (t<ti*w1+w2/2))
        ii  = i[idx]
        for n in range(N_e):
            fr[n] = sum(ii == n)
        dec = decode(fr, N_e)
        decs.append(dec)

    return decs, n_wins

def len2(x):
        if type(x) is not type([]):
                if type(x) is not type(np.array([])):
                        return -1
        return len(x)

def phase2(x):
        if not np.isnan(x):
                return cmath.phase(x)
        return np.nan

def circdist(angles1,angles2):
        if len2(angles2) < 0:
                if len2(angles1) > 0:
                        angles2 = [angles2]*len(angles1)
                else:
                        angles2 = [angles2]
                        angles1 = [angles1]             
        if len2(angles1) < 0:
                angles1 = [angles1]*len(angles2)
        return np.array(list(map(lambda a1,a2: phase2(np.exp(1j*a1)/np.exp(1j*a2)), angles1,angles2)))


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array-value)).argmin()
    return idx


def normgauss(xxx,sigma):
    gauss = (1/(sigma*np.sqrt(2*np.pi)) *np.exp(-(xxx-0)**2 / (2*sigma**2)))
    return gauss/gauss.max()


def normgrad(xxx):
    return np.gradient(xxx)/np.gradient(xxx).max()


def dog1(sigma,x):
    xxx     = np.arange(-200, 200, .1)
    dog_1st = normgrad(normgauss(xxx,sigma))
    return np.array(list(map(lambda x: dog_1st[find_nearest(xxx,x)], x)))


def rep_transform(prev,curr):
    global repstrength
    dist = np.degrees(circdist(np.radians(curr),np.radians(prev)))
    rep  = repstrength*dog1(45,dist) # or any other value that works
    print curr, curr - rep
    return curr - rep


def run_simulation(i):

    #np.seed(sd)
    global par, rep
    log_file    = "simulation_%i_%f_%s_%i" %(os.getpid(), time.time(), socket.gethostname(), i)
    beh_log     = "read_out_log_beh_geefact%.5f_rep%.2f.txt" %(par, repstrength)
    fr_log      = "read_out_log_fr_geefact%.5f_rep%.2f.txt" %(par, repstrength)
    stp_log     = "read_out_log_stp_geefact%.5f_rep%.2f.txt" %(par, repstrength)

    print log_file

    N_e                 = 1024
    N_i                 = 256

    stim1_location      = random.randint(0,359)#180
    stim2a_location     = random.randint(0,359)
    stim2_location      = rep_transform(stim1_location,stim2a_location)#235

    defaultclock.dt     = 0.1 * ms
    t_stimulus1_start   = 1500 * ms
    t_stimulus_duration = 250 * ms
    t_stimulus1_end     = t_stimulus1_start + t_stimulus_duration
    t_delay1_duration   = 1000 * ms
    t_response_start    = t_stimulus1_end + t_delay1_duration
    t_response_end      = t_response_start + t_stimulus_duration
    t_iti_duration      = 3000 * ms
    t_stimulus2_start   = t_response_end + t_iti_duration
    t_stimulus2_end     = t_stimulus2_start + t_stimulus_duration
    t_delay2_duration   = 3000 * ms

    sim_time            = t_stimulus2_end + t_delay2_duration

    # rate_monitor_excit, spike_monitor_excit, voltage_monitor_excit, idx_monitored_neurons_excit, synapse_monitor_excit, \
    # rate_monitor_inhib, spike_monitor_inhib, voltage_monitor_inhib, idx_monitored_neurons_inhib, synapse_monitor_inhib, weight_profile\
    spike_monitor_excit, synapse_monitor_excit \
	= simulate_wm(N_excitatory=N_e, N_inhibitory=N_i, 
		      stimulus_strength=.08 * namp, Jpos_excit2excit=1.63,
		      stimulus1_center_deg=stim1_location, stimulus2_center_deg=stim2_location, 
		      t_stimulus1_start=t_stimulus1_start, t_stimulus2_start=t_stimulus2_start,
		      t_stimulus_duration=t_stimulus_duration,
		      t_delay1=t_delay1_duration, t_delay2=t_delay2_duration, 
		      t_iti_duration=t_iti_duration, sim_time=sim_time)


    i,t         = spike_monitor_excit.it
    stp         = synapse_monitor_excit.stp
    # decs, nwins = readout(i, t, sim_time)

    fr_center   = []
    for n in range(int(sim_time/ms)-250):
        fr_center.append(4*sum((i[(t>=n*ms) & (t<n*ms+250*ms)] >= int(np.round(N_e/360. * stim1_location))-10) \
                         & (i[(t>=n*ms) & (t<n*ms+250*ms)] <= int(np.round(N_e/360. * stim1_location))+10))/21.)


    stp_center = np.mean(stp,axis=0)

    fr_delay = np.zeros(N_e)
    fr_iti   = np.zeros(N_e)

    fr_0sec  = np.zeros(N_e)
    fr_1sec  = np.zeros(N_e)
    fr_3sec  = np.zeros(N_e)
    for n in range(N_e):
	fr_delay[n] = 4*sum(i[(t>=t_response_start-250*ms) & (t<t_response_start)] == n)
	fr_iti[n]   = 4*sum(i[(t>=t_stimulus2_start-250*ms) & (t<t_stimulus2_start)] == n)
	fr_0sec[n]  = 4*sum(i[(t>=t_stimulus2_end) & (t<t_stimulus2_end+250 * ms)] == n)
	fr_1sec[n]  = 4*sum(i[(t>=t_stimulus2_end+750*ms) & (t<t_stimulus2_end+1000*ms)] == n)
	fr_3sec[n]  = 4*sum(i[(t>=t_stimulus2_end+2750*ms) & (t<t_stimulus2_end+3000*ms)] == n)
    dec_0sec   = decode(fr_0sec, N_e)
    dec_1sec   = decode(fr_1sec, N_e)
    dec_3sec   = decode(fr_3sec, N_e)


    with open(beh_log, 'a') as myfile:
	myfile.write(log_file+' '+str(np.around(np.radians(stim1_location),4)) \
	    +" "+str(np.around(np.radians(stim2a_location),4))+" "+str(dec_0sec)+" "+str(dec_1sec)+" "+str(dec_3sec)+'\n')

    with open(fr_log, 'a') as myfile:
	myfile.write(log_file)
	for fr in fr_center:
	    myfile.write(' '+str(np.around(fr,4)))
	myfile.write('\n')

    with open(stp_log, 'a') as myfile:
	myfile.write(log_file)
	for stp in stp_center:
	    myfile.write(' '+str(np.around(stp,4)))
	myfile.write('\n')
    print 'saved'


#####################################################################################################
#                                    RUN SIMULATIONS                                                #
#####################################################################################################

#Parallel(n_jobs=numcores)(delayed(run_simulation)(i) for i in range(numcores))


pool = mp.Pool(processes=numcores, maxtasksperchild=1)   
myseeds = range(0, numcores)
args = [(sd) for sd in myseeds]
pool.map(run_simulation, args, chunksize=1)
#

print 'all sims finished'
print time.time() - start_time 
