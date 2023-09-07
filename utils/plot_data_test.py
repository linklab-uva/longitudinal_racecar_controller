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
from scipy.ndimage import gaussian_filter1d

current_path = os.path.dirname(__file__)

#put your relative path to the csv files
raptor_msgs = pd.read_csv(os.path.join(current_path, '../data/lvms_brake_test_raptor_msgs.csv'))
imu_msgs = pd.read_csv(os.path.join(current_path, '../data/lvms_brake_test_imu_inputs.csv'))

def normalize_data(input_times, data):
    input_times = input_times[~np.isnan(data)]
    data = data[~np.isnan(data)]
    return (input_times - input_times[0]), data

parser = argparse.ArgumentParser(prog='Longitudinal Analyzer', description='Graphs n stuff', epilog='Made by John Link')


parser.add_argument('-w', '--vehicle-width', action='store', default=2.0, type=float)

args = parser.parse_args()

KPH2MPH = 0.621371
# print("The data options are")
# print(list(imu_msgs.columns))
# print(list(raptor_msgs.columns))

raptor_times = np.array(raptor_msgs['__time'])
imu_times = np.array(imu_msgs['__time'])


rear_right_wheel_speed = np.array(raptor_msgs['/raptor_dbw_interface/wheel_speed_report/rear_right'])
rear_left_wheel_speed = np.array(raptor_msgs['/raptor_dbw_interface/wheel_speed_report/rear_left'])
rear_avg_wheel_speed = (rear_right_wheel_speed + rear_left_wheel_speed) / 2
desired_vel = np.array(imu_msgs['/vehicle/desired_velocity_readout/data'])
wheel_speed_times, rear_avg_wheel_speed = normalize_data(raptor_times, rear_avg_wheel_speed * KPH2MPH)
des_vel_times, desired_vel = normalize_data(imu_times, desired_vel)

filtered_rear_avg_wheel_speed = gaussian_filter1d(rear_avg_wheel_speed, 10)

f2 = plt.figure(2)
f2.set_figwidth(10)
f2.set_figheight(5)
ax = plt.axes()
ax.scatter(wheel_speed_times, rear_avg_wheel_speed, s=.5, alpha=0.5, c='orange', label="Rear Avg Velocity")
ax.plot(wheel_speed_times, filtered_rear_avg_wheel_speed, 'b', label="Filtered Velocity")
ax.plot(des_vel_times, desired_vel, 'r', label="Desired Velocity")
ax.legend()
ax.set_ylabel("Speed (MPH)")
ax.set_xlabel("Time (Seconds)")
f2.show()

plt.show()
exit()

times = np.array(ben['__time'])
times = times[~np.isnan(gps_lat)]

times = times - times[0]

print(list(lap_times))

f2 = plt.figure(2)
f2.set_figwidth(5)
f2.set_figheight(5)
ax = plt.axes()
ax.scatter(local_coords[:,0], local_coords[:,1], c=local_coords[:, 3], s = 1, label='position')
ax.scatter(local_coords[300,0], local_coords[300,1], c=local_coords[300, 3], s = 100, label='position')

# for i, txt in enumerate(local_coords[:, 3]):
#     ax.annotate(i, (local_coords[i,0], local_coords[i,1]))
#     print(i)
plt.axis('equal')
plt.legend()
f2.show()
plt.show()