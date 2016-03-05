from ArgonSimulation import ArgonSimulation
import numpy as np
import sys
import time 

sim = ArgonSimulation()
sim.rescalingPeriod = 50
sim.rescalingTime = 30.0
sim.equilibrationTime = 40.0
sim.simulationTime = 60.0

Ms = [6]
Ts = [0.5]
rhos = np.concatenate((np.array([0.02, 0.05]), np.linspace(0.1, 2.0, 20)))
 
simJob = 5

rhoblock_start = int(sys.argv[1])
rhoblock_end = int(sys.argv[2])
rhos = rhos[rhoblock_start:rhoblock_end]

print("rhos for this simulation batch: " + str(rhos))

for M in Ms:
    for T in Ts:
        for rho in rhos:
            print("Starting simulation with parameters (sim nr " + str(simJob) + "): ")
            print("M:   " + str(M))
            print("T:   " + str(T))
            print("rho: " + str(rho))
            filename = "results/argon-longeq-M-%d-T-%.2f-rho-%.2f-simjob-%d.npz" % (M,T,rho,simJob)
            sim.initialise_arrays(T, rho, M)
            sim.do_simulation()
            sim.save_results(filename)