import numpy as np
from functions import MuonDetector

#%% creates a detector
depth = 0.75 # scintillator depth [cm]
length = 9 # scintillator length [cm]
sensor_number = 8 # amount of SiPM's
sensor_PDE = 0.38 # sensor photon detection efficiency
scintillator = "EJ-200" # scintillator type

detector = MuonDetector(depth, length, sensor_number, sensor_PDE, scintillator)
#%% creates a muon impact
muon_energy = 1000 # MeV
muon_x,muon_y = 3,6 # muon impact position [cm]
muon_angle = 0 # angle muon makes W.R.T the scintillator 0 is straight down [degrees]


photons_measured = detector.impact(muon_energy, muon_x, muon_y, muon_angle)
print(f"The measured photons at each sensor is: {photons_measured}")
#%% reconstructs the muon position and plots it
mux,muy = detector.fitting(photons_measured)
Fitting_plot = True # when set to false, the reconstructed position will not be shown.

print(f"The actual position is: x = {np.round(muon_x,2)}, y = {np.round(muon_y,2)}")
print(f"The reconstructed position is: x = {np.round(mux,2)}, y = {np.round(muy,2)}")

detector.plotting(Fitting_plot)

#%% calculates the uncertainty
iterations = 1000 # how many times to iterate to find the standard deviation

sigma_x,sigma_y = detector.uncertainty(iterations)

print(f"The uncertainty is: x = {np.round(sigma_x,2)}, y = {np.round(sigma_y,2)}")


