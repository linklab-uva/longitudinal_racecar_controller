import pandas as pd
import numpy as np
import os
import pygeodesy
from math import trunc
import matplotlib.pyplot as plt

current_path = os.path.dirname(__file__)

#put your relative path to the csv files
raptor_msgs = pd.read_csv(os.path.join(current_path, '../data/lvms_brake_test_raptor_msgs.csv'))
imu_msgs = pd.read_csv(os.path.join(current_path, '../data/lvms_brake_test_imu_inputs.csv'))



print("The data options are")
print(list(raptor_msgs.columns))
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