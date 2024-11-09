"""Variant of the Adaptive Exponential Leaky Integrate and Fire Model.
a 2-variable model than can reproduce many types of neural behavior
see Naud, Marcille, Clopath, Gerstner, Biol. Cybern. 2006

This code was adapted from the original Matlab code used in:
An Introductory Course in Computational Neuroscience 
by Paul Miller, Brandeis University (2017)

This code loops through different applied currents.
The adaptation term is altered in three ways:

1) It is a conductance rather than a current.
2) It has a time constant of 0.5ms, so acts as a refractory,
rate-limiting term rather than a standard spike-rate adaptation term.
3) A tan function of voltage is used, so the conductance increases to
very high levels if the membrane potential is high.

The tan function is so steep that it can counteract excessively strong
input currents that would otherwise cause the mean membrane potential to
constantly grow with input. This prevents the firing rates from increasing
without end.

The threshold term is altered as in Method 2 of Tutorial 2.2, reaching a
peak then decaying back to baseline following each spike.
The reset potential is set to the baseline of the threshold, so plays a 
greater role at higher rates when threshold is reached before it has 
returned to baseline.

A dip in membrane potential following reset arises from the refractory
conductance.
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
G_L = 10e-9               # Leak conductance (S)
C = 100e-12               # Capacitance (F)
E_L = -75e-3              # Leak potential (V)
E_K = -80e-3              # Leak potential (V)
V_Thresh_base = -60e-3    # Baseline of threshold potential (V)
Vmax = 150e-3             # Level of voltage to detect a spike
V_Thresh_peak = 4 * Vmax  # Maximum of threshold, post-spike
deltaT = 10e-3            # Threshold shift factor (V)
V_Reset = -60e-3          # Reset potential (V)

tau_g = 0.5e-3            # Adaptation time constant (s)
tau_vth = 2e-3

a = 10e-9                 # Adaptation recovery (S)
b = 60e-9                 # Adaptation strength (A)

I0 = 0e-9                 # Baseline current (A)

dt = 5e-7                 # Time-step in sec
tmax = 1                  # Maximum time in sec
tvector = np.arange(0, tmax, dt)  # Vector of all the time points

ton = 0                   # Time to switch on current step
toff = tmax               # Time to switch off current step
non = int(ton / dt)       # Index of time vector to switch on
noff = int(toff / dt)     # Index of time vector to switch off
I = np.full(tvector.shape, I0)  # Baseline current added to all time points

Iappvec = np.arange(0, 101, 10) * 1e-12  # List of applied currents

initialrate = np.zeros_like(Iappvec)
finalrate = np.zeros_like(Iappvec)
singlespike = np.zeros_like(Iappvec)
meanV = np.zeros_like(Iappvec)

# Loop through trials with different applied currents
for trial, Iapp in enumerate(Iappvec):
    I[non:noff] = Iapp  # Update with new applied current
    v = np.zeros_like(tvector)
    v[0] = E_L          # Initial membrane potential
    G_sra = np.zeros_like(tvector)
    spikes = np.zeros_like(tvector)
    V_Thresh = np.full(tvector.shape, V_Thresh_base)

    for j in range(len(tvector) - 1):
        if v[j] > Vmax + V_Thresh[j]:  # Spike detection
            v[j] = V_Reset
            G_sra[j] += b
            spikes[j] = 1
            V_Thresh[j] = V_Thresh_peak

        v[j + 1] = v[j] + dt * (G_L * (E_L - v[j] + deltaT * np.exp((v[j] - V_Thresh[j]) / deltaT))
                                + (E_K - v[j]) * G_sra[j] + I[j]) / C
        v[j + 1] = max(v[j + 1], E_K)

        V_Thresh[j + 1] = V_Thresh[j] + (V_Thresh_base - V_Thresh[j]) * dt / tau_vth
        Gss = a * np.tan(0.25 * np.pi * min((v[j] - E_K) / (V_Thresh_base + Vmax - E_K), 1.99999))
        G_sra[j + 1] = G_sra[j] + dt * (Gss - G_sra[j]) / tau_g
        print('[%d] ' % trial, 'Gss:', Gss)

    spiketimes = dt * np.where(spikes == 1)[0]

    if len(spiketimes) > 1:
        ISIs = np.diff(spiketimes)
        initialrate[trial] = 1 / ISIs[0]
        finalrate[trial] = 1 / ISIs[-1] if len(ISIs) > 1 else 0
    elif len(spiketimes) == 1:
        singlespike[trial] = 1

    meanV[trial] = np.mean(v)

# Plot the results
print("Plotting")
plt.figure(figsize=(10, 8))

plt.subplot(4, 1, 1)
plt.plot(tvector[:int(0.2 / dt)], v[:int(0.2 / dt)] * 1e3, 'k')
plt.xlabel('Time (sec)')
plt.ylabel('V_m (mV)')

plt.subplot(4, 1, 2)
plt.plot(Iappvec * 1e9, finalrate, 'k')
plt.xlabel('I_app (nA)')
plt.ylabel('Spike Rate (Hz)')

plt.subplot(4, 1, 3)
plt.plot(Iappvec * 1e9, meanV * 1e3, 'k')
plt.xlabel('I_app (nA)')
plt.ylabel('Mean V_m (mV)')

plt.subplot(4, 1, 4)
plt.plot(finalrate, meanV * 1e3, 'k')
plt.xlabel('Spike rate (Hz)')
plt.ylabel('Mean V_m (mV)')

plt.tight_layout()
plt.show()
