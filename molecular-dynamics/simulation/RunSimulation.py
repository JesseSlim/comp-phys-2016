from ArgonSimulation import ArgonSimulation
import numpy as np
import scipy.io as sc
import time

## simulation parameters
simulationTime = 10.0 # total simulation time
timeStep = 0.004 # time step
numFrames = 50 # number of frames of the molecular configuration to be shown during simulation
reportFreq = 0.01 # fraction of simulation progress that should be reported by a short text message

rescalingPeriod = 10 # number of time steps between each velocity rescaling to reach the desired temperature
rescalingTime = 1.0 # time until we should do the rescaling procedure to end up at the desired temperature
equilibrationTime = 1.0 # time until we consider the system to be in equilibrium

## system parameters
M = 5 # number of unit cells to initialize atoms in

desiredT = 3.0 # temperature the system should be equilibrated towards
rho = 0.8 # density of the system

record_data = False # do we save the data in an array or not

sim = ArgonSimulation()

Temp = 0.1;
dTemp = 0.1;
Dens = 0.1;
dDens = 0.1;
cv = np.zeros([20,40]);
i = 0;
j = 0;
while (Dens <= 2):
    while (Temp <= 4):
        sim.initialise_arrays(Temp, Dens, M)
        sim.do_simulation()
        print(Temp)
        print(Dens);
        cv[j,i] = sim.result_cv();
        Temp += dTemp
        Temp = np.round(Temp,3);
        i = i + 1;
    Dens += dDens
    Dens = np.round(Dens,3);
    Temp = 0.1;
    i = 0;
    j = j + 1;
sc.savemat('data'+str(time.strftime("%d-%m-%y_%H:%M:%S"))+'.mat', mdict={'cv':cv});