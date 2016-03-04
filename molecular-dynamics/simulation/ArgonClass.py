## init classes

import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import sys
from IPython import display

def print_fl(output):
    print(output)
    sys.stdout.flush()

@jit(nopython=True)
def calculate_quantities_jit(rIn, LIn, distHistBinsIn, distHistBinSizeIn):
    # apply periodic boundary conditions here
    nIn = rIn.shape[0]
    distOut = np.zeros((nIn, nIn))
    FOut = np.zeros((nIn, 3))
    distHistOut = np.zeros((distHistBinsIn))
    UOut = 0.0
    virialOut = 0.0
    epsilon = 1e-12 # fail safe in case distances between atoms become zero
    # loop through all of the atom pairs once
    for i in range(0,nIn):
        for j in range(0,i):
            # calculate distance
            dx = rIn[i,0] - rIn[j,0]
            dx -= np.around(dx/LIn) * LIn

            dy = rIn[i,1] - rIn[j,1]
            dy -= np.around(dy/LIn) * LIn

            dz = rIn[i,2] - rIn[j,2]
            dz -= np.around(dz/LIn) * LIn

            dr = np.sqrt(dx*dx + dy*dy + dz*dz)
            if dr == 0.0:
                # use failsafe small value for r in case the distance is zero
                # to prevent division by zero in the force calculation
                # note that the simulation will fail anyway probably as this
                # will introduce a huge potential energy but just in case
                dr = epsilon
            distOut[i,j] = distOut[j,i] = dr

            # put distance in distance histogram
            distHistOut[int(dr/distHistBinSizeIn)] += 1

            # calculate potential, force and virial, using the following intermediate result
            # for computational efficiency (2x speed-up)
            dr_6 = dr**6            
            UOut += 4.0*(1/(dr_6*dr_6) - 1/dr_6) 

            fLJ = (48.0/(dr_6*dr_6*dr) - 24.0/(dr_6*dr))
            virialOut += dr*fLJ

            f_x = dx / dr * fLJ
            f_y = dy / dr * fLJ
            f_z = dz / dr * fLJ

            FOut[i,0] += f_x
            FOut[i,1] += f_y
            FOut[i,2] += f_z

            FOut[j,0] += -f_x
            FOut[j,1] += -f_y
            FOut[j,2] += -f_z
    # return all the wonderfull things we've calculated
    return distOut, FOut, distHistOut, UOut, virialOut

class ArgonClass:
    
    def __init__(self, simulationTime, timeStep, numFrames, reportFreq, rescalingPeriod, rescalingTime, equilibrationTime,
                M, desiredT, rho, save):
        
        ## simulation parameters
        self.simulationTime = simulationTime # total simulation time
        self.h = timeStep # time step
        self.numSteps = int(self.simulationTime/self.h) # number of simulation steps
        self.numFrames = numFrames # number of frames of the molecular configuration to be shown during simulation
        self.reportFreq = reportFreq # fraction of simulation progress that should be reported by a short text message

        self.rescalingPeriod = rescalingPeriod # number of time steps between each velocity rescaling to reach the desired temperature
        self.rescalingTime = rescalingTime # time until we should do the rescaling procedure to end up at the desired temperature
        self.equilibrationTime = equilibrationTime # time until we consider the system to be in equilibrium
        self.equilibriumStart = int(self.equilibrationTime/self.h) # time step from which we consider the system to be in equilibrium

        ## system parameters
        self.M = M # number of unit cells to initialize atoms in

        self.desiredT = desiredT # temperature the system should be equilibrated towards
        self.rho = rho # density of the system

        self.n = 4*M**3 # total number of atoms to be simulated
        self.sigma_v = np.sqrt(desiredT) # spread of the initial velocities
        self.L = np.power(self.n/self.rho, 1/3) # linear size of the (cubic) system
        
        self.save = save # Determines whether we save the data to an array or not.
    
    def Init(self):
        ## output data
        self.t = 0 # initialize simulation time to zero
        if (self.save == True):
            self.v = np.zeros((self.numSteps,self.n,3)) # matrix to hold the 3 velocity components per atom at every time step
            self.r = np.zeros((self.numSteps,self.n,3)) # matrix to hold the 3 spatial coordinates per atom at every time step
            self.rNoPBC = np.zeros((self.numSteps,self.n,3)) # matrix to hold the 3 spatial coordinates per atom to which the PBC have *not* been applied (for diffusion)
            self.v_hat = np.zeros((self.numSteps,self.n,3)) # matrix to hold intermediate velocities during simulation (specific to the Verlet algorithm)
        else:
            self.v = np.zeros((self.n,3)) # matrix to hold the 3 velocity components per atom at every time step
            self.r = np.zeros((self.n,3)) # matrix to hold the 3 spatial coordinates per atom at every time step
            self.rNoPBC = np.zeros((self.n,3)) # matrix to hold the 3 spatial coordinates per atom to which the PBC have *not* been applied (for diffusion)
            self.v_hat = np.zeros((self.n,3)) # matrix to hold intermediate velocities during simulation (specific to the Verlet algorithm)
            
            
        self.dist = np.zeros((self.numSteps,self.n,self.n)) # matrix to hold the distances between all atoms at every time step
        self.F = np.zeros((self.numSteps,self.n,3)) # matrix to hold the forces between all atoms at every time step

        # keep track of a distance histogram to calculate the correlation function from
        self.distHistBins = 200
        self.distHistBinSize = np.sqrt(3)*self.L/(2*self.distHistBins)
        self.distHist = np.zeros((self.numSteps, self.distHistBins))

        # matrices to hold potential and kinetic energies at every time step
        self.U = np.zeros((self.numSteps))
        self.K = np.zeros((self.numSteps))
        # matrix to hold the value of the virial coefficient at every time step
        self.virial = np.zeros((self.numSteps))
        
    def Set(self, var, val):
        if hasattr(self, var):
            self.var = val;
            self.Init();
            
    def Get(self, var):
        if hassatt(self, str(var)):
            return self.var
            
    def initialise(self, set_v = True):
        t = 0;
        i = 0;
        L = self.L; M = self.M; t = self.t; r = self.r
        for mx in range(0, M):
            for my in range(0, M):
                for mz in range(0, M):
                    if (self.save == True):
                        r[t,i,0] = L*mx/M; r[t,i,1] = L*my/M; r[t,i,2] = L*mz/M
                        i += 1
                        r[t,i,0] = L*(mx+0.5)/M; r[t,i,1] = L*(my+0.5)/M; r[t,i,2] = L*mz/M
                        i += 1
                        r[t,i,0] = L*(mx+0.5)/M; r[t,i,1] = L*my/M; r[t,i,2] = L*(mz+0.5)/M
                        i += 1
                        r[t,i,0] = L*mx/M; r[t,i,1] = L*(my+0.5)/M; r[t,i,2] = L*(mz+0.5)/M
                        i += 1
                    else:
                        r[i,0] = L*mx/M; r[i,1] = L*my/M; r[i,2] = L*mz/M
                        i += 1
                        r[i,0] = L*(mx+0.5)/M; r[i,1] = L*(my+0.5)/M; r[i,2] = L*mz/M
                        i += 1
                        r[i,0] = L*(mx+0.5)/M; r[i,1] = L*my/M; r[i,2] = L*(mz+0.5)/M
                        i += 1
                        r[i,0] = L*mx/M; r[i,1] = L*(my+0.5)/M; r[i,2] = L*(mz+0.5)/M
                        i += 1
        if (self.save == True):
            self.rNoPBC[t,:,:] = r[t,:,:]
        else:
            self.rNoPBC[:,:] = r[:,:]
        self.r = r;

        if set_v:
            if (self.save == True):
                self.v[t,:,:] = np.random.normal(0.0, self.sigma_v, size=(self.n,3))
                self.v[t,:,:] = (self.v[t,:,:] - np.mean(self.v[t,:,:], axis=0)) / np.std(self.v[t,:,:]) * self.sigma_v
            else:
                self.v[:,:] = np.random.normal(0.0, self.sigma_v, size=(self.n,3))
                self.v[:,:] = (self.v[:,:] - np.mean(self.v[:,:], axis=0)) / np.std(self.v[:,:]) * self.sigma_v
        else:
            self.v[t,:,:] = 0.0
        if (self.save == True):
            self.K[t] = 0.5*np.sum(self.v[t,:,:]*self.v[t,:,:])
        else:
            self.K[t] = 0.5*np.sum(self.v[:,:]*self.v[:,:])
    
    def initialise_simple():
        self.M = 3; M = 3
        self.n = 2; n = 2
        t = self.t

        self.v = np.zeros((numSteps,n,3))
        self.r = np.zeros((numSteps,n,3))
        self.v_hat = np.zeros((numSteps,n,3))

        self.L = 3; L = 3;

        self.dist = np.zeros((numSteps,n,n))
        self.F = np.zeros((numSteps,n,3))

        self.v[t,:,:] = 0.0
        self.r[t,0,:] = [self.L-0.55*np.power(2.0,1.0/6.0), L/2, L/2]
        self.r[t,1,:] = [0.55*np.power(2.0,1.0/6.0), L/2, L/2]
        
    @jit()
    def update_coordinates(self):
        if (self.save == True):
            self.v_hat[self.t,:,:] = self.v[self.t,:,:] + self.h * self.F[self.t,:,:]/2
            self.r[self.t + 1,:,:] = (self.r[self.t,:,:] + self.h * self.v_hat[self.t,:,:]) % self.L
            self.rNoPBC[self.t + 1,:,:] = (self.rNoPBC[self.t,:,:] + self.h * self.v_hat[self.t,:,:])

            # calculate all the relevant quantities for the simulation
            self.t += 1
            self.calculate_quantities()

            self.v[self.t,:,:] = self.v_hat[self.t-1,:,:] + self.h*self.F[self.t,:,:] / 2
            self.K[self.t] = 0.5*np.sum(self.v[self.t,:,:]*self.v[self.t,:,:])
        else:
            print('test');
            self.v_hat[:,:] = self.v[:,:] + self.h * self.F[:,:]/2
            self.r[:,:] = (self.r[:,:] + self.h * self.v_hat[:,:]) % self.L
            self.rNoPBC[:,:] = (self.rNoPBC[:,:] + self.h * self.v_hat[:,:])

            # calculate all the relevant quantities for the simulation
            self.t += 1
            self.calculate_quantities()

            self.v[:,:] = self.v_hat[:,:] + self.h*self.F[:,:] / 2
            self.K[self.t] = 0.5*np.sum(self.v[:,:]*self.v[:,:])
    
    def calculate_quantities(self):
        # we can't use globals inside a jitted function, so we provide this little wrapper function
        if (self.save == True):
            r = self.r; t = self.t; L = self.L; distHistBins = self.distHistBins; distHistBinSize = self.distHistBinSize
            self.dist[self.t,:,:], self.F[self.t,:,:], self.distHist[self.t,:], self.U[self.t], self.virial[self.t] = calculate_quantities_jit(r[t,:,:], L, distHistBins, distHistBinSize)
        else:
            r = self.r; t = self.t; L = self.L; distHistBins = self.distHistBins; distHistBinSize = self.distHistBinSize
            self.dist[:,:], self.F[:,:], self.distHist[self.t,:], self.U[self.t], self.virial[self.t] = calculate_quantities_jit(
            r[:,:], L, distHistBins, distHistBinSize)
        
    def do_simulation(self, vis = False, out = False):
        if (vis):
            self.start_visualisation()

        self.initialise()
        self.calculate_quantities()
        while(self.t < self.numSteps - 1):
            if (self.t*self.h < self.rescalingTime and self.t > 0 and self.t % self.rescalingPeriod == 0):
                self.scale_to_temperature()
            if (self.t % int(self.numSteps*self.reportFreq) == 0 and out ):
                print_fl("Simulation progress: " + str(int(self.t*100/self.numSteps)) + "%")
            if (self.t % int(self.numSteps/self.numFrames) == 0 and vis):
                self.visualize()   
            self.update_coordinates()
        if (vis):
            self.end_visualisation()
        if (out):
            print_fl("Simulation finished")
        
    def start_visualisation(self):
        
        self.fig = None
        self.ax = None
        
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
    
    def end_visualisation(self):
        display.clear_output()
        
    def scale_to_temperature(self):
        # average the velocities over the second half of the last rescaling period
        halfRP = int(self.rescalingPeriod/2)
        lbda = np.sqrt((self.n - 1)*3*self.desiredT*halfRP/(2*np.sum(self.K[self.t-halfRP:self.t])))

        self.v[self.t,:,:] *= lbda
        
    def visualize(self):
        self.ax.cla()
        self.ax.scatter(self.r[self.t,:,0], self.r[self.t,:,1], self.r[self.t,:,2])
        self.ax.set_xlim((0.0, self.L))
        self.ax.set_ylim((0.0, self.L))
        self.ax.set_zlim((0.0, self.L))
        display.clear_output(wait=True)
        display.display(self.fig)
        #time.sleep(0.05)
        
    def getCV(self):
        delta_K_sq = np.var(self.K[self.equilibriumStart:])
        K_sq = np.mean(self.K[self.equilibriumStart:])**2
        C_v = 3 * K_sq / (2*K_sq - 3*self.n*delta_K_sq)
        return C_v