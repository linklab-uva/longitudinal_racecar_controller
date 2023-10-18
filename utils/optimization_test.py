from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize
import numpy as np
import pandas as pd
import numpy as np
from math import trunc
import matplotlib.pyplot as plt
import argparse, sys, os
import pygeodesy
from scipy.signal import butter, filtfilt, resample
from scipy.spatial.distance import cdist
from scipy.interpolate import make_interp_spline, make_smoothing_spline, BSpline
from scipy.ndimage import uniform_filter1d, spline_filter1d
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.optimize import curve_fit


current_path = os.path.dirname(__file__)
data = pd.read_csv(os.path.join(current_path, '../data/exported_data.csv'))
VARS = (
    data['rpm'].to_numpy(),
    data['gear'].to_numpy(),
    data['throttle'].to_numpy(),
    data['velocity'].to_numpy(),
    data['front_brake_pressure'].to_numpy(),
    data['rear_brake_pressure'].to_numpy()
)
observed_accel = data['expected_accel'].to_numpy()
observed_accel_filtered = gaussian_filter1d(observed_accel, 10)

torque_matrix = np.loadtxt(os.path.join(current_path, 'dyno_matrix_maker/new_data/output_matrix.csv'), delimiter=',')
# torque_matrix_np = torque_matrix.to_numpy()


gear_ratios = np.array([0, 2.9167, 1.875, 1.3809, 1.1154, 0.96, 0.8889])
final_drive = 3.0

def acceleration_on_car(VARS, c1, c2, a1, a2, a3, b1):
    rpm, gear, thr, v, bkpf, bkpr = VARS
    filtered_rpm = rpm.astype(int) - 1600
    filtered_rpm[filtered_rpm < 0] = 0

    et = (c1 * torque_matrix[filtered_rpm, thr.astype(int)] * final_drive * gear_ratios[gear.astype(int)]) / 0.3113

    engine_decel = (c2 * final_drive * gear_ratios[gear.astype(int)]) / 0.3113
    engine_decel[np.logical_or(et > 250, thr < 3)] = 0
    engine_decel[rpm < 1500] = 0

    et[rpm < 1600] = 0.0
    et[thr < 3] = 0.0
    
    brake_decel = (b1 * gaussian_filter1d(bkpf + bkpr, 10)) / 0.3113
    brake_decel[v < 0.5] = 0
    rolling = np.ones_like(filtered_rpm) * a3
    rolling[v < 0.5] = 0
    
    return (et - brake_decel - (v**2 * a1) - engine_decel)/750 - rolling


def objective(params):
    # Your code to compute the objective
    return np.sum((observed_accel_filtered - acceleration_on_car(VARS, *params)) ** 2)

kernel = C(1.0, (1e-3, 1e5)) * RBF(10, (1e-2, 1e3))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)


# Assume 6 parameters in your model
tries = 200
initial_points = np.column_stack([np.random.rand(tries) * .4 + .55, np.random.rand(tries) * 50 + 40, np.random.rand(tries) * 0.3 + 0.1, np.random.rand(tries), np.random.rand(tries), np.random.rand(tries) * 0.2 + .3])  # 10 initial random points, change dimensions accordingly
initial_objective = np.array([objective(params) for params in initial_points])

gp.fit(initial_points, initial_objective)


def acquisition_function(params):
    params = params.reshape(1, -1)
    mu, sigma = gp.predict(params, return_std=True)
    return -mu + 0.01 * sigma  # Change the exploration factor as needed


for i in range(50):  # Number of optimization iterations
    next_point = minimize(acquisition_function, [0.65, 40, 0.4, 0.00, 0.59, 0.3], options={'maxiter': 10000}).x
    next_objective = objective(next_point)
    gp.fit(np.vstack([initial_points, next_point]), np.hstack([initial_objective, next_objective]))



optimal_params = initial_points[np.argmin(initial_objective)]

print(optimal_params)


predicted_accel_redefined = acceleration_on_car(VARS, *optimal_params)

original_predicted_accel_redefined = acceleration_on_car(VARS, *[0.85, 40, 0.4, 0.00, 0.59, 0.4])

# Calculate residuals with the new parameters
residuals_redefined = observed_accel_filtered - predicted_accel_redefined

# Create the plots
fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Plot real vs predicted acceleration
axs[0].plot(observed_accel_filtered, label='Filtered Real Acceleration', color='blue', linewidth= 0.5)
axs[0].plot(predicted_accel_redefined, label='Predicted Acceleration', color='red', linewidth= 0.5)
axs[0].plot(original_predicted_accel_redefined, label='Predicted Acceleration', color='orange', linewidth= 0.5)
axs[0].set_title('Filtered Real vs Predicted Acceleration (Redefined Function)')
axs[0].set_xlabel('Index')
axs[0].set_ylabel('Acceleration')
axs[0].legend()

# Plot residuals
axs[1].plot(residuals_redefined, label='Residuals', color='green')
axs[1].set_title('Residuals (Redefined Function)')
axs[1].set_xlabel('Index')
axs[1].set_ylabel('Residuals')
axs[1].legend()
for ax in axs:
    ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)

plt.tight_layout()
plt.show()