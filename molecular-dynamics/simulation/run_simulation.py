from ArgonSimulation import ArgonSimulation
import numpy as np
import sys
import time 

sim = ArgonSimulation()

Ms = [6]
Ts = np.linspace(0.0, 4.0, 17)
rhos = np.concatenate((np.array([0.02, 0.05]), np.linspace(0.1, 2.0, 20)))

simJob = 5

Tsblock_start = int(sys.argv[1])
Tsblock_end = int(sys.argv[2])

Ts = Ts[Tsblock_start:Tsblock_end]

print("Ts for this simulation batch: " + str(Ts))

for M in Ms:
    for T in Ts:
        for rho in rhos:
            print("Starting simulation with parameters (sim nr " + str(simJob) + "): ")
            print("M:   " + str(M))
            print("T:   " + str(T))
            print("rho: " + str(rho))
            filename = "results/argon-M-%d-T-%.2f-rho-%.2f-simjob-%d.npz" % (M,T,rho,simJob)
            sim.initialise_arrays(T, rho, M)
            sim.do_simulation()
            sim.save_results(filename)