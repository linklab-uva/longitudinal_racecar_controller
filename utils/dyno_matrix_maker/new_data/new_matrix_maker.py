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


torque_40 = np.array(pd.read_csv(os.path.join(os.path.realpath(os.path.dirname(__file__)), "40_torque_data.csv"), header=None, sep=',', skipinitialspace = True))
torque_50 = np.array(pd.read_csv(os.path.join(os.path.realpath(os.path.dirname(__file__)), "50_torque_data.csv"), header=None, sep=',', skipinitialspace = True))
torque_60 = np.array(pd.read_csv(os.path.join(os.path.realpath(os.path.dirname(__file__)), "60_torque_data.csv"), header=None, sep=',', skipinitialspace = True))
torque_65 = np.array(pd.read_csv(os.path.join(os.path.realpath(os.path.dirname(__file__)), "65_torque_data.csv"), header=None, sep=',', skipinitialspace = True))
torque_70 = np.array(pd.read_csv(os.path.join(os.path.realpath(os.path.dirname(__file__)), "70_torque_data.csv"), header=None, sep=',', skipinitialspace = True))
torque_75 = np.array(pd.read_csv(os.path.join(os.path.realpath(os.path.dirname(__file__)), "75_torque_data.csv"), header=None, sep=',', skipinitialspace = True))
torque_80 = np.array(pd.read_csv(os.path.join(os.path.realpath(os.path.dirname(__file__)), "80_torque_data.csv"), header=None, sep=',', skipinitialspace = True))
torque_100 = np.array(pd.read_csv(os.path.join(os.path.realpath(os.path.dirname(__file__)), "100_torque_data.csv"), header=None, sep=',', skipinitialspace = True))

torque_40[:, 0] *= 1000
torque_50[:, 0] *= 1000
torque_60[:, 0] *= 1000
torque_65[:, 0] *= 1000
torque_70[:, 0] *= 1000
torque_75[:, 0] *= 1000
torque_80[:, 0] *= 1000
torque_100[:, 0] *= 1000

for item in range(10):
    torque_40[:,0][item::10] += item
    torque_50[:,0][item::10] += item
    torque_60[:,0][item::10] += item
    torque_65[:,0][item::10] += item
    torque_70[:,0][item::10] += item
    torque_75[:,0][item::10] += item
    torque_80[:,0][item::10] += item
    torque_100[:,0][item::10] += item
    


y_vals = int(1e2) + 1
torque_matrix = np.zeros((7600 - 1600, y_vals))
valid_thr = [40, 50, 60, 65, 70, 75, 80, 100]
torque_matrix.fill(np.nan)

torque_matrix[2900 - 1601 : 7339 - 1600,  40] = torque_40[:, 1]
torque_matrix[3000 - 1601 : 7499 - 1600,  50] = torque_50[:, 1]
torque_matrix[3000 - 1601 : 7499 - 1600,  60] = torque_60[:, 1]
torque_matrix[3080 - 1601 : 7499 - 1600,  65] = torque_65[:, 1]
torque_matrix[3120 - 1601 : 7479 - 1600,  70] = torque_70[:, 1]
torque_matrix[3080 - 1601 : 7489 - 1600,  75] = torque_75[:, 1]
torque_matrix[3000 - 1601 : 7499 - 1600,  80] = torque_80[:, 1]
torque_matrix[3240 - 1601 : 7499 - 1600,  100] = torque_100[:, 1]




b,a = butter(1,0.003, 'low')
b2,a2 = butter(1,0.1, 'low')

f2 = plt.figure(2)
f2.set_figwidth(10)
f2.set_figheight(10)
ax = plt.axes(projection='3d')
# ax = plt.figure().add_subplot(projection='3d')
for item in valid_thr:
    data = torque_matrix[:, item]
    rpm = np.arange(1600, 7600)
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
    rpm = np.arange(1600, 7600)
    data = torque_matrix[:, item]
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

for rpm in range(7600 - 1600):

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
plt.plot(absc, s1(absc), label="6900")
plt.plot(absc, s2(absc), label="4900")
plt.legend()
plt.show()
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