import serial
import numpy as np
import matplotlib.pyplot as plt

from functions import MuonDetector 
#%%

depth = 0.75
side_length = 9
sensor_number = 4

detector = MuonDetector(depth,side_length,sensor_number,sensor_QE = 0.38,scintillator = "EJ-200")


#%%

COM1 = 'COM5'
COM2 = 'COM6'
COM3 = 'COM7'
COM4 = 'COM8'

ser1 = serial.Serial(COM1)
ser2 = serial.Serial(COM2)
ser3 = serial.Serial(COM3)
ser4 = serial.Serial(COM4)

data1 = np.array([])
data2 = np.array([])
data3 = np.array([])
data4 = np.array([])

photons_measured = np.array([])

count = 0

muonposx,muonposy = [] , []

try:
    while True:
        try:
            line1 = float(ser1.readline().decode().strip('\r\n'))
        except UnicodeDecodeError:
            line1 = 0
            
        try:
            line2 = float(ser2.readline().decode().strip('\r\n'))*1.074618204837391
        except UnicodeDecodeError:
            line2 = 0
            
        try:
            line3 = float(ser3.readline().decode().strip('\r\n'))*1.1051675030516748
        except UnicodeDecodeError:
            line3 = 0 
        try:
            line4 = float(ser4.readline().decode().strip('\r\n'))*1.4122010398613518
        except UnicodeDecodeError:
            line4 = 0
        
        if line1 >= 12:
        
            count += 1 
       
            data1 = np.append(data1, line1)
            data2 = np.append(data2, line2)
            data3 = np.append(data3, line3)
            data4 = np.append(data4, line4)
        
            

        
            print(f"{COM1}:" , line1)
            print(f"{COM2}:" , line2)
            print(f"{COM3}:" , line3)
            print(f"{COM4}:" , line4)
            
            
            try:
                mux,muy =  detector.fitting(np.array([line1,line2,line3,line4]))
                plt.scatter(mux,muy, marker="x", s = 40, c = 'C1', label = f'Reconstructed position at:  {round(float(mux),2),round(float(muy),2)}')
                plt.gca().set_aspect('equal')
                plt.grid()
                plt.xlabel("x (cm)"), plt.ylabel("y (cm)")
                plt.xlim(0, side_length), plt.ylim(0, side_length)
                plt.xticks([0, 1/4 * side_length , 1/2 * side_length , 3/4 * side_length , side_length])
                plt.yticks([0, 1/4 * side_length , 1/2 * side_length , 3/4 * side_length , side_length])
                plt.legend(bbox_to_anchor =(0.85,-0.15))
                
                muonposx.append(mux)
                muonposy.append(muy)
                
                plt.pause(0.25)
                plt.show()
            except:
                print("Warning: Muon position not found")
                pass
        
except KeyboardInterrupt: # CTRL+C in command line ends the program.
    ser1.close()
    ser2.close()
    ser3.close()
    ser4.close()


# %%

counts, xedges, yedges, im = plt.hist2d(muonposx, muonposy, cmap='plasma', bins = int(side_length)*5, range=[[0,side_length],[0,side_length]])
cbar = plt.colorbar(im)
cbar.set_label('amount of detections', rotation=270, labelpad = 15)
plt.gca().set_aspect('equal')
plt.grid()
plt.xlim(0, side_length), plt.ylim(0, side_length)
plt.xlabel("x (cm)"), plt.ylabel("y (cm)")
plt.xticks([0, 1/4 * side_length , 1/2 * side_length , 3/4 * side_length , side_length])
plt.yticks([0, 1/4 * side_length , 1/2 * side_length , 3/4 * side_length , side_length])
plt.title(f'Total measured hits: {count}')
plt.show()




