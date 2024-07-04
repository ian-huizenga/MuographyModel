import random
import pylandau
import numpy as np
import numpy.random as r
from typing import Literal
from scipy.stats import poisson
import scipy.constants as const
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# can be found at https://pdg.lbl.gov/2023/AtomicNuclearProperties/index.html p is k in the dictionary
scintvals = {"EJ-200"   :{    "tag": "organic", # wether scintillator is organic or inorganic
                               "x0": 0.1464, # density effect parameters
                               "x1": 2.4855,
                               "d0": 0,
                               "C" : 3.1997, 
                               "a" : 0.16101,
                               "p" : 3.2393, # "k" in link
                               "I" : 64.7E-6, # MeV
                               "rho": 1.032, # density, g/cm^3
                               "Z/A":0.54141, # atomic number/atomic mass
                               "L0": 400, # light attenuation length
                               "n1" : 1.52, # ref index
                               "kb" : 1.26E-2, # birks' constant
                               "Gain": 9200 # photons/MeV

                           
                        },
             "NaI"      :{     "tag": "inorganic", # wether scintillator is organic or inorganic
                               "x0": 0.1203, # density effect parameters
                               "x1": 3.5920,    
                               "d0": 0,
                               "C" : 6.0572,
                               "a" : 0.12516,
                               "p" : 3.0398, # "k" in link
                               "I" : 452E-6, # MeV
                               "rho": 3.667, # density, g/cm^3
                               "Z/A": 0.42697, # atomic number/atomic mass
                               "L0" : 258.8, # light attenuation length
                               "n1" : 1.775, # ref index
                               "Gain":40000, # photons/MeV
                               "c1" : 1.08,  # extra fitting parameters since Birks' law only works on organic
                               "c2" : 1.65E-3, # sadly can't find the source for these, so take these with a grain of salt
                               "c3" : 0.594, # :(
                                 
                 }}

class Calculations:
    def __init__(self, depth,side_length, scintillator:str): 
        """
        Class for all the calculations, starting with a muon hit and losing energy,
        to a final amount of light measured at each sensor.

        Parameters
        ----------
        depth : Float. Thickness of the scintillator. For accurate results it 
        should lie in the range of 0.1 - 1 cm
        
        side_length : Float. Length of the side of the scintillator, I.E if
        the scintillator is square, it is simply the suqare root of the surface
        area.
        
        scintillator : String. Which type of scintillator is used. More can be
        added if appropriate data is found and implemented according to the 
        dictionary in this document.

        Raises
        ------
        ValueError
            If scintillator type is not in the dictionary.

        Returns
        -------
        None.

        """
        self.scintillator = scintvals[scintillator]        
        if scintillator not in scintvals.keys():
            raise ValueError("Scintillator not found.")
    
    def dEdx(self, E):
        
        """
        Bethe bloch equation, calculates mean ionisation energy losses for
        muons through a scintillator. 
        
        A scintillator can be added in the dictionary.
        In theory, all plastic scintillators are compatible, 
        given that their data can be found/approximated.
        
        Although there is a tag for inorganic scintillators, this is merely for
        Testing purposes and should not be used in its current state.
        
        Parameters
        ----------
            E : Float. Muon kinetic energy [MeV], most muons are around 1-4 GeV.
        
        Returns
        -------
            dEdx_avg : Float. Mean energy loss [MeV/g/cm^2].
            Should be on the order of 1-5 MeV/g/cm^2 for the average kinetic energy.
        """

        scintillator = self.scintillator
        (x0,x1,d0,C,a,p,I,rho,ZA) = list(scintillator.values())[1:10]  


        mass = 105.6583755 #muon mass MeV/c^2
        z = 1 # muon charge (technically negative but the square gets taken anyway)

        m = mass # idk why i did this tbh
        beta = np.sqrt(1-(1/(1+E/m))**2) # just v/c but this way i can work with MeV
        gamma = 1 + E/m # standard lorentz factor just written differently
        mec = 0.511 #MeV /c ^2
        
        x = np.log10(beta*gamma)  # this is just a term thats used in the bethe bloch
        
        delta = 0
        
        if x < x0:
            delta = d0*10**(2*(x-x0))
        elif x0 <= x < x1:
            delta = 2*np.log(10)*x - C + a*(x1-x)**p
        elif x > x1:
            delta = 2*np.log(10)*x - C

        # max transferrable kinetic energy from muon to electron
        Tmax = (2 * mec * beta**2 * gamma**2) / (1+2*gamma*(mec/m) + (mec**2/m**2))
        
        k = 0.307075
        b = z**2 * ZA * 1/beta**2
        d = 1/2*np.log((2 * mec * beta**2 * gamma**2 * Tmax) / I**2) - beta**2 - delta / 2 + 1/8 * Tmax**2/(gamma*m*const.c**2)
        
        
        dEdx = k*b*d
        return dEdx
    
    def landau(self,E):
        """
        The landau distribution relating to the average energy loss found with
        the bethe bloch equation. 

        Parameters
        ----------
        E : Float. Muon kinetic energy [MeV], most muons are around 1-4 GeV.

        Returns
        -------
        dEdx : Array. PDF of the landau distribution.

        """
        
        scintillator = self.scintillator
        mean = self.dEdx(E) # mean energy loss used to calc MPV by pylandau.
        ZA = list(scintillator.values())[9]
        
        mass = 105.6583755 #muon mass MeV/c^2
        
        x = np.linspace(0, 10, int(1000)) # array for the PDF
    
        beta = np.sqrt(1-(1/(1+E/mass))**2)
        
        k = 0.307075/2 * ZA * 1/beta**2 
        
        eta = k*4 # width of landau
        return pylandau.landau_pdf(x,mean,eta)
    

    def dLdx(self,E):
        """
        Birks' law. Calculates amount of photons generated per MeV/g/cm^2.
        Although both organic and inorganic scintillators can be used, it is
        unknown how trustworthy the inorganic tag is. Therefore, unless this 
        gets researched. It is recommended that this is not used.
        
        Takes a random number according to the PDF generated by the Landau 
        distribution and uses this as its' energy loss.
        
        Parameters
        ----------
        E : Float. Muon kinetic energy [MeV], most muons are around 1-4 GeV.

        Returns
        -------
        dLdx : Random float. Amount of photons generated per g/cm^2 of distance.

        """
        
        scintillator = self.scintillator
        landau_distr = self.landau(E)
        
        # Random val from landau
        dEdx = np.asarray(random.choices(np.linspace(0, 10, int(1000)),landau_distr))
        
        # checks wether scintillator is organic or inorganic
        match scintillator["tag"]:
            case "organic":
                (kb,S) = list(scintillator.values())[-2:]
                return S * dEdx /(1 + kb  *dEdx)
            case "inorganic": # Todo: research this
                (S,c1,c2,c3) = list(scintillator.values())[-4:]
                return c1/(1 + c2*dEdx + c3*(dEdx)**-1) * S
            
    def N0(self,E, distance_traversed):
        """
        Total amount of photons generated with muon traversal through scintillator.

        Parameters
        ----------
        E : Float. Muon kinetic energy [MeV], most muons are around 1-4 GeV.
        distance_traversed : Float. Distance [cm], this is the distance that 
        muon traverses through the scintillator. 

        Returns
        -------
        N0: Total amount of photons [-] generated during scintillation.

        """
        scintillator = self.scintillator
        dLdx = self.dLdx(E)
        density = list(scintillator.values())[8]
        return dLdx*density*distance_traversed
    
    def N(self,depth,side_length):
        """
        Calculates how many photons hit the sensor, formatted this way
        due to scipy.curve_fit.
        
        Parameters
        ----------
        depth : Float. Thickness of the scintillator. For accurate results it 
        should lie in the range of 0.1 - 1 cm
        
        side_length : Float. Length of the side of the scintillator, I.E if
        the scintillator is square, it is simply the suqare root of the surface
        area.
        
        Returns
        -------
        N : Function used in curve fitting and calculating how many photons land
        on the sensor.

        """
        def Ncurry(XY, xmu,ymu,N0):
            """
            Actual function used to determine light yield at sensors after
            a muon traverses through the scintillator. 

            Parameters
            ----------
            XY : Numpy 2D array. Array of x and y coordinates, are handled
            automatically by the get_sensor_locations() function after a number
            of sensors is specified.
            
            xmu : Float. Muon x position [cm].
            
            ymu : Float. Muon y position [cm].
            
            N0 : Float. Total amount of photons [-] generated during scintillation.

            Returns
            -------
            N : Numpy Ndarray. Total amount of photons measured at each sensor.

            """
            scintillator = self.scintillator
            L0 = list(scintillator.values())[10]
            n1 = list(scintillator.values())[11]
            
            x,y = XY
            
            L = side_length
            l = depth
            n2 = 1.000273 # ref index air
            thetc = np.arcsin(n2/n1)
            
            R = np.hypot((xmu-x),(ymu-y))
            phi = np.zeros(len(x))
            
            
            for i in range(len(phi)):
                d1 = np.transpose(np.array([0,0]))
                d2 = np.transpose(np.array([float(x[i]-xmu),float(y[i]-ymu)]))
                
                if x[i] == 0:
                    d1[0] = 1
                elif x[i] == L:
                    d1[0] = -1
                else: d1[0] = 0
                    
                if y[i] == 0:
                    d1[1] = 1
                elif y[i] == L:
                    d1[1] = -1
                else: d1[1] = 0
                
                
                phi[i] = np.abs(np.dot(d1, d2)/( np.sqrt( d2[0]**2 + d2[1]**2)*np.sqrt(d1[0]**2+d1[1]**2)))
            
            return N0*(np.cos(thetc))*(1+ 2*R/L0 * (1 - 1/np.sin(thetc))) * (1/np.pi * np.arctan(l/(2*R)*phi))
        return Ncurry
        
                
                
    

class MuonDetector(Calculations):

    def __init__(self,depth:float,side_length:float,sensor_number: Literal[4,8,16,32],sensor_PDE,scintillator):
        """
        Class that uses all the functions from the Calculation class to simulate
        a muon hit through a scintillator.
               

        Parameters
        ----------
        depth : Float. Thickness of the scintillator. For accurate results it 
        should lie in the range of 0.1 - 1 cm
        
        side_length : Float. Length of the side of the scintillator, I.E if
        the scintillator is square, it is simply the suqare root of the surface
        area.
        
        sensor_number : Literal[4,8,16,32]. Total number of sensors used,
        other values are not implemented
        
        sensor_PDE : Float. Sensor photon detection efficiency (PDE)
            
        scintillator : String. Which type of scintillator is used. More can be
        added if appropriate data is found and implemented according to the 
        dictionary in this document.

        Raises
        ------
        ValueError
            If sensor number is not  4,8,16 or 32.

        Returns
        -------
        None.

        """
        if sensor_number not in [4,8,16,32]:
            raise ValueError("Value must be 4,8,16 or 32.")
        
        super().__init__(depth,side_length,scintillator)
        
        self.sensor_PDE = sensor_PDE
        self.depth = depth
        self.side_length = side_length
        self.sensor_number = sensor_number
        self._get_sensor_locations()
        
    def _get_sensor_locations(self):
        """
        Function to set sensor coordinates based on sensor number. 

        Returns
        -------
        sx : List. Sensor x coordinates.
        
        sy : List. Sensor y coordinates.

        """
        # code is ugly as hell but hey it works, could not find a better way of
        # doing it sadly.
        
        side_length = self.side_length
        sensor_number = self.sensor_number
        
        if sensor_number == 4:
            sx = [1/2*side_length,  1/2*side_length,       0,      side_length]
            sy = [    0,      side_length,   1/2*side_length,  1/2*side_length]
            
        elif sensor_number == 8: 
            sx = [1/2*side_length,  1/2*side_length,       0,      side_length,             0,              0,           side_length,            side_length]
            sy = [    0,      side_length,   1/2*side_length,  1/2*side_length,             0,              side_length,           0,            side_length]
        
        elif sensor_number == 16:
            sx = [1/2*side_length,             1/2*side_length,             0,           side_length,             0,              0,           side_length,            side_length,0,0,1/4*side_length,3/4*side_length,1/4*side_length,3/4*side_length,side_length,side_length]
            sy = [0,                        side_length,       1/2*side_length,      1/2*side_length,             0,            side_length,             0,            side_length,1/4*side_length,3/4*side_length,0,0,side_length,side_length,1/4*side_length,3/4*side_length]

        elif sensor_number == 32:
            sx = [1/2*side_length,             1/2*side_length,             0,           side_length,             0,              0,           side_length,            side_length,0,0,1/4*side_length,3/4*side_length,1/4*side_length,3/4*side_length,side_length,side_length, 1/8*side_length,3/8*side_length,5/8*side_length,7/8*side_length,1/8*side_length,3/8*side_length,5/8*side_length,7/8*side_length,0,0,0,0,side_length,side_length,side_length,side_length]
            sy = [0,                        side_length,       1/2*side_length,      1/2*side_length,             0,            side_length,             0,            side_length,1/4*side_length,3/4*side_length,0,0,side_length,side_length,1/4*side_length,3/4*side_length,0,0,0,0,side_length,side_length,side_length,side_length,1/8*side_length,3/8*side_length,5/8*side_length,7/8*side_length,1/8*side_length,3/8*side_length,5/8*side_length,7/8*side_length]
            
        else:
            sx = np.nan
            sy = np.nan
            
        self.sx = sx
        self.sy = sy
        
        
    def impact(self,E:float, muon_x:float, muon_y:float, muon_angle:float):
        """
        Simulates a muon impact through a scintillator according to the physics
        behind scintillation and energy loss. Takes sensor PDE and poisson
        statistics into account

        Parameters
        ----------
        E : Float. Muon kinetic energy [MeV], most muons are around 1-4 GeV.
        
        xmu : Float. Muon x position [cm].
        
        ymu : Float. Muon y position [cm].
        
        muon_angle : Float. Muon incident angle [degrees].

        Returns
        -------
        photons_measured : Total amount of photons measured at each sensor.

        """
        
        self.muon_x = muon_x
        self.muon_y = muon_y 
        self.E = E
        self.muon_angle = muon_angle
        
        sensor_number = self.sensor_number
        
        depth = self.depth
        side_length = self.side_length
        distance_traversed = depth/np.cos(np.deg2rad(muon_angle))
        N0 = self.N0(E, distance_traversed)
        
        XY = np.array([self.sx,self.sy])
        try:
            N = poisson.rvs(self.N(depth,side_length)(XY,muon_x,muon_y,N0))
        except ValueError:
            N = np.ones(sensor_number)
        
        photons_measured = np.zeros(len(self.sx))
        
        for i in range(len(photons_measured)):
            photons_measured[i] = sum(r.choice([0, 1], N[i].astype(int), p=[1-self.sensor_PDE, self.sensor_PDE]))
        
        
        return photons_measured
    
    def fitting(self, photons_measured = None):
        """
        Fitting algorithm, uses the main equation to curve fit xmu, ymu
        and N_0. N_0 is not given as an output as it is not relevant for this. 
        

        Parameters
        ----------
        photons_measured : Optional. Numpy array of length sensors. 
        Total amount of photons measured by each SiPM. The measured voltage of 
        a SiPM can also be used as an input. 
        In this case, N_0 will become V_0. xmu and ymu stay the same. 
        
        If photons_measured is not given, it will calculate based on the impact()
        function

        Returns
        -------
        xmu : Float. Fitted muon x position.
        ymu : Float. Fitted muon y position.

        """
        
        if photons_measured is None: # if not given, it will attempt to use impact()
        
        
            muon_x = self.muon_x
            muon_y = self.muon_y
            E = self.E
            muon_angle= self.muon_angle
            photons_measured = self.impact(E, muon_x, muon_y, muon_angle)
      
        
      
        depth = self.depth
        side_length = self.side_length
        XY = np.array([self.sx,self.sy])
        

        N = self.N(depth,side_length)
        
        
        try:
            (xmu,ymu,No) , _ = curve_fit(N, XY, photons_measured/self.sensor_PDE, bounds=((0,0,0),(side_length,side_length,20000000)))
        except RuntimeError: # if it can't be found, return nan
            xmu = np.nan
            ymu = np.nan
            print("Warning: Could not find impact location")
        return xmu,ymu
    
    def plotting(self, fitting_plot:bool = True):
        """
        Plotting function. Simple function to display the scintillator,
        actual muon hit, and fitted muon hit.

        Parameters
        ----------
        fitting_plot : Optional. Bool. If set to False, it will not show the fitted
        muon impact location. Default is True

        Returns
        -------
        None.

        """
        
        side_length = self.side_length
        plt.scatter(self.sx,self.sy,marker="s",s = 40, label = "Sensor")
        
        try:
            plt.scatter(self.muon_x,self.muon_y, marker="x", s = 40, c = "#ff007f", label = f'Muon impact at: {round(float(self.muon_x),2),round(float(self.muon_y),2)}')
        except:
            print("Warning: Not plotting an impact, perhaps you put plotting() before impact()?")
            pass
        
        if fitting_plot == True:
            try:
                mux,muy =  self.fitting()
                plt.scatter(mux,muy, marker="x", s = 40, c = 'C1', label = f'Reconstructed position at:  {round(float(mux),2),round(float(muy),2)}')
            except:
                print("Warning: Muon position not found")
                pass
        
        plt.gca().set_aspect('equal')
        plt.grid()
        plt.xlabel("x (cm)"), plt.ylabel("y (cm)")
        plt.xlim(0, side_length), plt.ylim(0, side_length)
        plt.xticks([0, 1/4 * side_length , 1/2 * side_length , 3/4 * side_length , side_length])
        plt.yticks([0, 1/4 * side_length , 1/2 * side_length , 3/4 * side_length , side_length])
        plt.legend(bbox_to_anchor =(0.85,-0.15))
       
        
        plt.show()
        
    def uncertainty(self, iterations:int, Heatmap:bool = True, Print_uncertainty: bool = False, Raw_uncertainty: bool = True):
        """
        Function to calculate positional uncertainty, iterates N times to find
        the standard deviation in both x and y. Generates a heatmap of all
        fitted positions, can be used to, relatively easily find the positional
        uncertainty of a required detector.

        Parameters
        ----------
        iterations : int. How many times the function needs to iterate to calculate
        a positional uncertainty. 
        
        Heatmap : Optional. Bool. If set to false the function will not plot 
        the heatmap. The default is True.
        
        Print_uncertainty : Optional. Bool. If set to false the function will not print
        the uncertainty. The default is True.
        
        Raw_uncertainty : Optional. Bool. If set to false the function will not return
        the uncertainty. The default is True.

        Returns
        -------
        sigma : Optional. Tuple. If Raw_uncertainty is set to True, this is the 
        positional uncertainty (sigma_x,sigma_y).

        """
        
        side_length = self.side_length
        muxlist = np.zeros(iterations)
        muylist = np.zeros(iterations)
        for i in range(len(muxlist)):
            muxlist[i],muylist[i] =  self.fitting()
        
        sigmax,sigmay = np.nanstd(muxlist),np.nanstd(muylist)
        
        if Heatmap == True:
            counts, xedges, yedges, im = plt.hist2d(muxlist, muylist, cmap='plasma', bins = int(side_length)*5, range=[[0,side_length],[0,side_length]])
            plt.scatter(self.sx,self.sy,marker="s",s = 40, label = "Sensor")
            try:
                plt.scatter(self.muon_x,self.muon_y, marker="x", s = 40, c = "#ff007f", label = f'Muon impact at: {round(float(self.muon_x),2),round(float(self.muon_y),2)}')
            except:
                print("Warning: Not plotting an impact, perhaps you put iterations() before impact()?")
                pass
            cbar = plt.colorbar(im)
            cbar.set_label('amount of detections', rotation=270, labelpad = 15)
            plt.plot([], [], ' ', label=f'Uncertainty is x: {np.round(sigmax,2)} cm, y: {np.round(sigmay,2)} cm')
            plt.gca().set_aspect('equal')
            plt.grid()
            plt.xlim(0, side_length), plt.ylim(0, side_length)
            plt.xlabel("x (cm)"), plt.ylabel("y (cm)")
            plt.xticks([0, 1/4 * side_length , 1/2 * side_length , 3/4 * side_length , side_length])
            plt.yticks([0, 1/4 * side_length , 1/2 * side_length , 3/4 * side_length , side_length])
            plt.legend(bbox_to_anchor =(1,-0.15))
            plt.title('Simulation')
            plt.savefig("Improvedposdetsim2.pdf", format = "pdf", bbox_inches="tight")
            plt.show()
        if Print_uncertainty == True:
            print(f'Uncertainty is x: {np.round(sigmax,3)} cm, y: {np.round(sigmay,3)} cm')
        
        if Raw_uncertainty == True:
            sigma = (sigmax,sigmay)
            return sigma
       
        
        
if __name__ == "__main__":
    
    depth = 0.75
    side_length = 9
    sensor_number = 4
    
    muon_energy = r.normal(3000,500)#MeV
    
    muon_x = 3#r.uniform(0,side_length)
    muon_y = 2.3#r.uniform(0,side_length)
    muon_angle = r.normal(0,25)
    
    iterations = 95
    
    detector = MuonDetector(depth,side_length,sensor_number,sensor_PDE = 0.38,scintillator = "EJ-200")
    detector.impact(muon_energy, muon_x, muon_y, muon_angle)
    
    detector.uncertainty(iterations)

    
    