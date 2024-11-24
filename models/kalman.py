import numpy as np
import matplotlib.pyplot as plt

# Parameters for the simulation
dt = 0.1  # time step (s)
total_time = 20  # total time for the simulation (s)
n_steps = int(total_time / dt)

# True initial conditions
true_position = 0  # starting position (m)
true_velocity = 1  # constant velocity (m/s)

# Initial conditions for the Kalman filter
position_estimate = 0
velocity_estimate = 0
position_var = 1  # initial position variance
velocity_var = 1  # initial velocity variance

# Noise parameters
measurement_noise_std = 2  # standard deviation of measurement noise
process_noise_std = 0.1  # standard deviation of process noise

# State transition matrix
A = np.array([[1, dt], [0, 1]])
# Process noise covariance matrix
Q = np.array([[dt**4/4, dt**3/2], [dt**3/2, dt**2]]) * process_noise_std**2
# Observation matrix
H = np.array([[1, 0]])
# Measurement noise covariance
R = np.array([[measurement_noise_std**2]])

# Initialize arrays to store results
true_positions = []
measurements = []
kalman_positions = []
kalman_velocities = []
kalman_gains = []

# Initial state estimate and covariance
x_estimate = np.array([[position_estimate], [velocity_estimate]])
P = np.array([[position_var, 0], [0, velocity_var]])

# Simulation loop
for step in range(n_steps):
    # True state update (position and velocity)
    true_position += true_velocity * dt
    true_positions.append(true_position)
    
    # Simulated noisy measurement
    measurement = true_position + np.random.normal(0, measurement_noise_std)
    measurements.append(measurement)
    
    # Kalman filter prediction
    x_predict = A @ x_estimate
    P_predict = A @ P @ A.T + Q
    
    # Kalman gain
    S = H @ P_predict @ H.T + R
    K = P_predict @ H.T @ np.linalg.inv(S)
    
    # Measurement update
    y = measurement - (H @ x_predict)  # residual
    x_estimate = x_predict + K @ y
    P = (np.eye(2) - K @ H) @ P_predict
    
    # Store Kalman filter estimates
    kalman_positions.append(x_estimate[0, 0])
    kalman_velocities.append(x_estimate[1, 0])
    kalman_gains.append(K[0, 0])  # Store only the position component of the Kalman Gain

# Plotting the results
time = np.linspace(0, total_time, n_steps)

plt.figure(figsize=(12, 6))

# Plot true positions and measurements
plt.plot(time, true_positions, label="True Position", linestyle="--", color="blue")
plt.plot(time, measurements, label="Noisy Measurements", linestyle=":", color="red", alpha=0.6)

# Plot Kalman filter estimates
plt.plot(time, kalman_positions, label="Kalman Position Estimate", color="green")

plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.title("Kalman Filter Position Estimation")
plt.legend()
plt.grid()
plt.show()

# Velocity plot
plt.figure(figsize=(12, 6))
plt.plot(time, [true_velocity]*n_steps, label="True Velocity", linestyle="--", color="blue")
plt.plot(time, kalman_velocities, label="Kalman Velocity Estimate", color="green")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title("Kalman Filter Velocity Estimation")
plt.legend()
plt.grid()
plt.show()

# Plot Kalman Gain over time
plt.figure(figsize=(12, 6))
plt.plot(time, kalman_gains, label="Kalman Gain for Position Estimate", color="purple")
plt.xlabel("Time (s)")
plt.ylabel("Kalman Gain")
plt.title("Kalman Gain Over Time")
plt.legend()
plt.grid()
plt.show()
