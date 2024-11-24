"""Exercise 1.2 from Theoretical Neuroscience by Dayan and Abbott"
"""

import numpy as np
import matplotlib.pyplot as plt

# Function to generate Poisson spikes
def poisson_spike_generator_refractory(initial_rate, duration, dt=0.001, t_ref=0.002):
    """
    Generates a Poisson spike train with refractory period.

    Parameters:
        initial_rate (float): Average firing rate in Hz.
        duration (float): Duration of simulation in seconds.
        dt (float): Time step in seconds.
        t_ref (float): Refractory period in seconds.
    Returns:
        np.array: Array of spike times (in seconds).
    """
    rate = initial_rate
    num_steps = int(duration / dt)
    spike_times = []

    for t in range(num_steps):
        time = t * dt
        p_spike = rate * dt
        rand = np.random.rand()
        if rand < p_spike:
            # spike and reset rate to 0
            spike_times.append(time)
            rate = 0
        else:
            # recover exponentially
            rate += (dt / t_ref) * (initial_rate - rate)

    return spike_times

# Parameters
rate = 100  # Firing rate in Hz
duration = 10.0  # Duration in seconds
dt = 0.001  # Time step in seconds
t_ref = 0.002  # Refractory time in seconds

# Generate Poisson spikes
spike_times = poisson_spike_generator_refractory(rate, duration, dt, t_ref)

# Compute interspike intervals (ISIs)
interspike_intervals = np.diff(spike_times)

# Compute Coefficient of Variation (CV) of ISIs
cv = np.std(interspike_intervals) / np.mean(interspike_intervals)

# Compute Fano Factor
count_intervals = np.arange(0.001, 0.101, 0.001)  # Counting intervals from 1 ms to 100 ms
fano_factors = []
for interval in count_intervals:
    num_bins = int(duration / interval)
    spike_counts, _ = np.histogram(spike_times, bins=num_bins)
    fano_factors.append(np.var(spike_counts) / np.mean(spike_counts))


plt.figure(figsize=(10, 8))

# Plot spikes over time
plt.subplot(3, 1, 1)
plt.plot(spike_times, np.ones_like(spike_times), '|')
plt.xlabel('Time (s)')
plt.title('Poisson Spike Train with Refractory Period')
plt.yticks([])

# Plot ISI histogram
plt.subplot(3, 1, 2)
plt.hist(interspike_intervals, bins=50, density=True, alpha=0.7, color='blue')
plt.xlabel('Interspike Interval (s)')
plt.ylabel('Density')
plt.title('Interspike Interval Histogram')

# Plot Fano factor vs counting interval
plt.subplot(3, 1, 3)
plt.plot(count_intervals * 1000, fano_factors, marker='o', linestyle='-')
plt.xlabel('Counting Interval (ms)')
plt.ylabel('Fano Factor')
plt.title('Fano Factor vs Counting Interval')

plt.tight_layout()
plt.show()

# Output statistics
cv, fano_factors[:10]  # Show CV and first 10 Fano factors
