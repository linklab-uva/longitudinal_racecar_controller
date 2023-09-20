import pandas as pd
import numpy as np
import os
from scipy import interpolate
from matplotlib import cm
from math import isnan, nan
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, make_smoothing_spline, BSpline
from scipy.ndimage import gaussian_filter1d 
from scipy.signal import butter, filtfilt, savgol_filter

dataframe = pd.read_csv(os.path.join(os.path.realpath(os.path.dirname(__file__)), "iac_dyno_data.csv"))
dataframe.drop(dataframe.columns[[2, 4, 6, 8, 10, 12, 14]], axis=1, inplace=True)


x_vals = np.array(dataframe['RPM'])
y_vals = int(1e2) + 1
torque_matrix = np.zeros((7000 - 1600, y_vals))
valid_thr = [30, 50, 65, 70, 75, 80, 100]
torque_matrix.fill(np.nan)
# torque_matrix[:, 0] = dataframe['RPM']
torque_matrix[2518- 1601:, 30] = np.array(dataframe['30'])
torque_matrix[2518- 1601:, 50] = np.array(dataframe['50'])
torque_matrix[2518- 1601:, 65] = np.array(dataframe['65'])
torque_matrix[2518- 1601:, 70] = np.array(dataframe['70'])
torque_matrix[2518- 1601:, 75] = np.array(dataframe['75'])
torque_matrix[2518- 1601:, 80] = np.array(dataframe['80'])
torque_matrix[2518- 1601:, 100] = np.array(dataframe['100'])

b,a = butter(3,0.001, 'low')
b2,a2 = butter(5,0.1, 'low')

f2 = plt.figure(2)
f2.set_figwidth(10)
f2.set_figheight(10)
ax = plt.axes(projection='3d')
# ax = plt.figure().add_subplot(projection='3d')
for item in valid_thr:
    data = np.array(dataframe[f'{item}'])
    rpm = np.array(dataframe['RPM'])
    rpm = rpm[~np.isnan(data)]
    data = data[~np.isnan(data)]
    ax.scatter(item, rpm, data, s= 0.8)
ax.set_ylim(1600, 7000)
ax.set_xlim(0, 100)
ax.set_xlabel('Throttle Percent', fontsize=12, rotation=150)
ax.set_ylabel('Engine RPM', fontsize=12)
ax.set_zlabel('Torque (nm)', fontsize=12, rotation=60)

# plt.show()

for item in valid_thr:
    rpm = np.array(dataframe['RPM'])
    data = np.array(dataframe[f'{item}'])
    index = 0
    endindex = 0
    for idx, val in enumerate(data):
        if not isnan(val):
            index = idx
            break
    for idx, val in enumerate(np.flip(data)):
        if not isnan(val):
            endindex = len(data) - idx - 1
            break

    if index > 0:
        d1 = interpolate.interp1d([1300, rpm[index]], [0, data[index]], fill_value='extrapolate', kind='linear')
        arr1 = d1(np.arange(1600, rpm[index], 1))
        arr1[arr1 < 0] = 0.0
        torque_matrix[:rpm[index] - 1600, item] = arr1
    if endindex > 0:
        # print(rpm[endindex - 10 : endindex])
        # print(data[endindex - 10 : endindex])
        # d2 = interpolate.interp1d(rpm[endindex - 10 : 7000], data[endindex - 10 : endindex], fill_value=0.0, kind='nearest')
        # arr2 = d1(np.arange(rpm[endindex], 7000, 1))
        # arr2[arr2 < 0] = 0.0
        arr2 = np.ones(torque_matrix[rpm[endindex]-1600:, item].shape) * data[endindex]
        torque_matrix[rpm[endindex]-1600:, item] = arr2

for rpm in range(7000 - 1600):

    index = 0
    for idx, val in enumerate(torque_matrix[rpm, :]):

        if not isnan(val):
            index = idx
            break

    if index == 0:
        continue
    d1 = interpolate.interp1d([0, index], [0, torque_matrix[rpm, index]], fill_value='extrapolate', kind='linear')
    
    arr1 = d1(np.arange(0, index, 1))
    arr1[arr1 < 0] = 0.0
    torque_matrix[rpm, :index] = arr1


x = np.arange(0, torque_matrix.shape[1])
y = np.arange(0, torque_matrix.shape[0])
array = np.ma.masked_invalid(torque_matrix)
xx, yy = np.meshgrid(x, y)
#get only the valid values

x1 = xx[~array.mask]
y1 = yy[~array.mask]
newarr = array[~array.mask]

points = np.column_stack([x1, y1])

GD1 = interpolate.griddata((x1, y1), newarr, (xx, yy), method='linear', fill_value=0.0)

for idx in range(len(GD1[0])):

    # spline = make_smoothing_spline(torque_matrix[idx, :])
    GD1[:, idx] = filtfilt(b, a, GD1[:, idx])
rpm_splines = []
for idx in range(len(GD1)):

    GD1[idx, :] = filtfilt(b2, a2, GD1[idx, :])
    
absc = np.arange(0, 101)
s1 = make_smoothing_spline(absc, GD1[5300])
s2 = make_smoothing_spline(absc, GD1[3300])
# plt.plot(absc, s1(absc), label="6900")
# plt.plot(absc, s2(absc), label="4900")
# plt.legend()
# plt.show()
new_df = pd.DataFrame(GD1)

new_df.to_csv(os.path.join(os.path.realpath(os.path.dirname(__file__)), "output_matrix.csv"), sep=',', header=False, index=False)
f2 = plt.figure(2)
f2.set_figwidth(10)
f2.set_figheight(10)
ax = plt.axes(projection='3d')
# ax = plt.figure().add_subplot(projection='3d')

GD1[GD1 < 0] = 0.0

ax.plot_wireframe(xx + 1, yy + 1600, GD1, rstride=100, cstride=5, color='k', linewidth=0.8)

ax.set_xlabel('Throttle Percent', fontsize=12, rotation=150)
ax.set_ylabel('Engine RPM', fontsize=12)
ax.set_zlabel('Torque (nm)', fontsize=12, rotation=60)
# plt.axis('equal')
# ax.yaxis._axinfo['label']['space_factor'] = 3.0
# ax.plot_wireframe(xx + 1, yy + 1600, torque_matrix, rstride=100, cstride=10, color='k')

plt.show()