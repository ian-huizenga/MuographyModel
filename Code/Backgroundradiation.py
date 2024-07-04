"Simulation of the muon background radiation"

import numpy as np
import numpy.random as r
import matplotlib.pyplot as plt
from functions import MuonDetector

#%% creates a detector
depth = 0.75 # scintillator depth [cm]
length = 9 # scintillator length [cm]
sensor_number = 8 # amount of SiPM's
sensor_PDE = 0.38 # sensor photon detection efficiency
scintillator = "EJ-200" # scintillator type

detector = MuonDetector(depth, length, sensor_number, sensor_PDE, scintillator)
#%% creates a random muon impact at a random location and repeats this N times

N = 10000 # amount of random muon hits

positions = np.zeros([N,2])

for i in range(N):
    muon_x = r.uniform(0,length)
    muon_y = r.uniform(0,length)
    muon_angle = r.normal(0,25)
    muon_energy = 1000 # does not have a large effect on light yield, mainly landau
    
    detector.impact(muon_energy, muon_x, muon_y, muon_angle)
    positions[i] = detector.fitting()
#%% makes a heatmap of all reconstructed positions
# pattern that emerges at low sensor number comes from the viewing angle each sensor has.

counts, xedges, yedges, im = plt.hist2d(positions[:,0], positions[:,1], bins = int(length)*5, range=[[0,length],[0,length]])

cbar = plt.colorbar(im)
cbar.set_label('amount of detections', rotation=270, labelpad = 15)
plt.gca().set_aspect('equal')
plt.grid()
plt.xlim(0, length), plt.ylim(0, length)
plt.xlabel("x (cm)"), plt.ylabel("y (cm)")
plt.xticks([0, 1/4 * length , 1/2 * length , 3/4 * length , length])
plt.yticks([0, 1/4 * length , 1/2 * length , 3/4 * length , length])
plt.title(f'Total reconstructed hits: {np.count_nonzero(~np.isnan(positions[:,0]))}')
plt.show()

