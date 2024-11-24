import numpy as np
import matplotlib.pyplot as plt

# Define parameters and initial conditions
part = 'e'  # Options are 'b', 'c', 'd', 'e', 'f' for question part

dt = 2e-8          # Time step (s) must be extremely small
tmax = 0.35        # Maximum simulation time
t = np.arange(0, tmax, dt)  # Time vector

V_L = -0.060       # Leak reversal potential (V)
E_Na = 0.045       # Sodium channel reversal potential (V)
E_K = -0.082       # Potassium channel reversal potential (V)
V0 = -0.065

G_L = 30e-9        # Leak conductance (S)
G_Na = 12e-6       # Sodium conductance (S)
G_K = 3.6e-6       # Potassium conductance (S)

Cm = 100e-12       # Membrane capacitance (F)

istart = 100e-3    # Time applied current starts
ilength = 5e-3     # Length of applied current pulse
Ibase = 0e-9       # Baseline current
Npulses = 1        # Number of current pulses
pulsesep = 20e-3   # Separation between current pulses
V0 = -0.065        # Initial condition for V
m0 = 0.05          # Initial condition for m
h0 = 0.5           # Initial condition for h
n0 = 0.35          # Initial condition for n

# Set different parameters based on part
if part == 'b':
    ilength = 100e-3
    Ie = 0.22e-9
elif part == 'c':
    Npulses = 10
    Ie = 0.22e-9
    pulsesep = 18e-3
elif part == 'd':
    Npulses = 10
    Ibase = 0.6e-9
    Ie = 0e-9
elif part == 'e':
    Ibase = 0.65e-9
    Ie = 1e-9
elif part == 'f':
    Ibase = 0.65e-9
    Ie = 1e-9
    m0, h0, n0 = 0, 0, 0

# Set up the applied current vector
Iapp = Ibase * np.ones_like(t)
for pulse in range(Npulses):
    pulsestart = istart + pulse * pulsesep
    pulsestop = pulsestart + ilength
    Iapp[int(pulsestart / dt):int(pulsestop / dt)] = Ie

# Initialize simulation variables
V = np.full_like(t, V0)
n = np.full_like(t, n0)
m = np.full_like(t, m0)
h = np.full_like(t, h0)

Itot = np.zeros_like(t)
I_Na = np.zeros_like(t)
I_K = np.zeros_like(t)
I_L = np.zeros_like(t)
Vmax = max(V)

# Enable interactive plotting mode
plt.ion()

# Create figure and subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Initial plot setup for applied current
line1, = ax1.plot([], [], 'k')
ax1.set_ylabel('I_app (nA)')
ax1.set_xlim(0, tmax)
ax1.set_ylim(-0.5, 1.05)
ax1.set_title(f'4.1 Part {part}')

# Initial plot setup for membrane potential
line2, = ax2.plot([], [], 'k')
ax2.set_xlabel('Time (sec)')
ax2.set_ylabel('V_m (mV)')
ax2.set_xlim(0, tmax)
ax2.set_ylim(-85 if max(V) > 0 else -80, 45 if max(V) > 0 else -55)

# Simulation loop with live updating
for i in range(1, len(t)):
    Vm = V[i - 1]

    print('%d' % i)
    
    # Calculate alpha and beta for sodium and potassium gating variables
    alpha_m = (1e5 * (-Vm - 0.045)) / (np.exp(100 * (-Vm - 0.045)) - 1) if Vm != -0.045 else 1e3
    beta_m = 4000 * np.exp((-Vm - 0.070) / 0.018)
    alpha_h = 70 * np.exp(50 * (-Vm - 0.070))
    beta_h = 1000 / (1 + np.exp(100 * (-Vm - 0.040)))
    alpha_n = (1e4 * (-Vm - 0.060)) / (np.exp(100 * (-Vm - 0.060)) - 1) if Vm != -0.060 else 100
    beta_n = 125 * np.exp((-Vm - 0.070) / 0.08)

    tau_m = 1 / (alpha_m + beta_m)
    m_inf = alpha_m / (alpha_m + beta_m)
    tau_h = 1 / (alpha_h + beta_h)
    h_inf = alpha_h / (alpha_h + beta_h)
    tau_n = 1 / (alpha_n + beta_n)
    n_inf = alpha_n / (alpha_n + beta_n)
    
    # Update gating variables
    m[i] = m[i - 1] + (m_inf - m[i - 1]) * dt / tau_m
    h[i] = h[i - 1] + (h_inf - h[i - 1]) * dt / tau_h
    n[i] = n[i - 1] + (n_inf - n[i - 1]) * dt / tau_n
    
    # Calculate currents
    I_Na[i] = G_Na * m[i] ** 3 * h[i] * (E_Na - Vm)
    I_K[i] = G_K * n[i] ** 4 * (E_K - Vm)
    I_L[i] = G_L * (V_L - Vm)
    
    Itot[i] = I_L[i] + I_Na[i] + I_K[i] + Iapp[i]
    
    # Update membrane potential
    V[i] = V[i - 1] + Itot[i] * dt / Cm
    Vmax = max(Vmax, V[i])

    # Update plot every 5000 steps
    if i % 5000 == 0:
        line1.set_data(t[:i:10], 1e9 * Iapp[:i:10])   # Applied current (in nA)
        line2.set_data(t[:i:10], 1e3 * V[:i:10])      # Membrane potential (in mV)
        ax2.set_ylim(-85 if Vmax > 0 else -80, 45 if Vmax > 0 else -55)
        fig.canvas.draw()
        fig.canvas.flush_events()

# Finalize the plot
plt.ioff()
plt.show()
