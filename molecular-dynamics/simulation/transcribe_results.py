import numpy as np

Ms = [6]
Ts = np.linspace(0.25, 4.0, 16)
rhos = np.concatenate((np.array([0.02, 0.05]), np.linspace(0.1, 2.0, 20)))

simJobs = [1]

resultFiles = []

for M in Ms:
    for T in Ts:
        for rho in rhos:
        	for simJob in simJobs:
        		resultFiles.append("argon-M-%d-T-%.2f-rho-%.2f" % (M,T,rho))

for rFile in resultFiles:
	print("Processing file: " + rFile)
	with np.load("results-longrun/" + rFile + "-simjob-1.npz") as data:
		newData = dict()
		for key in data:
			if not (key == "distance_histogram"):
				newData[key] = data[key]

		# calculation of instantaneous temperature contained an error, recalculate (and recalculate pressure)
		# fixed as of 3-3-2016 22:30
		#newData["temperature"] = 2*data["kinetic_energy"]/(3*(data["n"] - 1))
		
		#newData["pressure"] = 1 + 1/(3 * data["n"] * np.mean(newData["temperature"])) * np.mean(data["virial"])
		np.savez("results-longrun-nohist/" + rFile + "-simjob-1.npz", **newData)