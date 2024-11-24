from brian2 import *
import matplotlib.pyplot as plt

# Parameters
num_neurons = 5
tau = 10 * ms
v_rest = -70 * mV
v_reset = -65 * mV
v_thresh = -50 * mV
refractory_period = 5 * ms

print(ms, mV)

# Neuron model
eqs = '''
dv/dt = (v_rest - v) / tau : volt
'''

# Create neuron group
neurons = NeuronGroup(
    num_neurons, model=eqs, threshold='v > v_thresh', reset='v = v_reset',
    refractory=refractory_period, method='exact'
)

# Initialize membrane potential
# neurons.v = v_rest
neurons.v = v_rest  # Random initial membrane potential

# Synaptic connections
poisson_input = PoissonGroup(N=10, rates=20 * Hz)  # 10 neurons firing at 20 Hz
syn = Synapses(poisson_input, neurons, on_pre='v += 10 * mV')  # Simple excitatory connection
syn.connect(condition='i != j', p=1)  # Connect with 10% probability

# Record data
spikemon = SpikeMonitor(neurons)
statemon = StateMonitor(neurons, 'v', record=True)

# Run simulation
run(1000 * ms, 'stdout')

# Plot spikes
plt.figure(figsize=(10, 4))
plt.subplot(211)
plt.plot(spikemon.t/ms, spikemon.i, 'k.')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')
plt.title('Spike raster plot')

# Plot membrane potentials
plt.subplot(212)
for i in range(num_neurons):
    plt.plot(statemon.t/ms, statemon.v[i]/mV, label=f'Neuron {i}')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane potential (mV)')
plt.legend()
plt.tight_layout()
plt.show()
