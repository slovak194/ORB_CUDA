import numpy as np


import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


track_times_fiel_path = "build/track_times_0.txt"

with open(track_times_fiel_path) as ff:
    track_times = np.fromfile(ff, sep='\n')

plt.plot(1/track_times, ".")
plt.plot(0, 0)

track_times_fiel_path = "build/track_times_1.txt"

with open(track_times_fiel_path) as ff:
    track_times = np.fromfile(ff, sep='\n')

plt.plot(1/track_times, ".")
plt.plot(0, 0)

# %%

# nvprof --export-profile timeline.prof app
#
# sudo nvprof -f --metrics achieved_occupancy,ipc -o metrics.prof test/test_perf
# sudo /usr/local/cuda-11.0/bin/nvvp -vm /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java ./metrics.prof


