import numpy as np

Ms = [6]
Ts = np.linspace(0.0, 4.0, 17)
rhos = np.concatenate((np.array([0.02, 0.05]), np.linspace(0.1, 2.0, 20)))

simJobs = np.arange(1,5)

resultFiles = []

for M in Ms:
    for T in Ts:
        for rho in rhos:
        	for simJob in simJobs:
        		resultFiles.append("argon-M-%d-T-%.2f-rho-%.2f-simjob-%d.npz" % (M,T,rho,simJob))

for rFile in resultFiles:
	print("Processing file: " + rFile)
	with np.load("results/" + rFile) as data:
		newData = dict()
		for key in data:
			if not key == "distance_histogram":
				newData[key] = data[key]
		np.savez("results-nohist/" + rFile, **newData)