import numpy as np


import matplotlib

matplotlib.use('Qt5Agg')
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


track_times_fiel_path = "build/track_times.txt"

with open(track_times_fiel_path) as ff:
    track_times = np.fromfile(ff, sep='\n')

plt.plot(1/track_times, ".")
plt.plot(0, 0)

# %%

# nvprof --export-profile timeline.prof app
#
# nvprof --metrics achieved_occupancy,ipc -o metrics.prof <app>


