import numpy as np
from numba import jit
import sys

def print_fl(output):
	print(output)
	sys.stdout.flush()

# numba doesn't support the jitting of class methods, so we provide this function
# outside of the main class and use 
@jit(nopython=True)
def calculate_quantities_jit(rIn, LIn, distHistBinsIn, distHistBinSizeIn):
	nIn = rIn.shape[0]
	distOut = np.zeros((nIn, nIn))
	FOut = np.zeros((nIn, 3))
	distHistOut = np.zeros((distHistBinsIn))
	UOut = 0.0
	virialOut = 0.0
	epsilon = 1e-5 # fail safe in case distances between atoms become zero
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



class ArgonSimulation:
	## simulation parameters
	simulationTime = 40.0 # total simulation time
	h = 0.004 # time step
	numSteps = None # number of simulation steps
	numFrames = 50 # number of frames of the molecular configuration to be shown during simulation
	reportFreq = 0.05 # fraction of simulation progress that should be reported by a short text message

	rescalingPeriod = 100 # number of time steps between each velocity rescaling to reach the desired temperature
	rescalingTime = 15.0 # time until we should do the rescaling procedure to end up at the desired temperature
	equilibrationTime = 20.0 # time until we consider the system to be in equilibrium
	equilibriumStart = None # time step from which we consider the system to be in equilibrium

	diffusionTrackTime = 20.0
	diffusionTrackSteps = None

	## system parameters
	M = None # number of unit cells to initialize atoms in

	desiredT = None # temperature the system should be equilibrated towards
	rho = None # density of the system

	n = None # total number of atoms to be simulated
	sigma_v = None # spread of the initial velocities
	L = None # linear size of the (cubic) system

	## output data
	t = 0 # initialize simulation time to zero
	v = None # matrix to hold the 3 velocity components per atom at every time step
	r = None # matrix to hold the 3 spatial coordinates per atom at every time step
	rNoPBC = None # matrix to hold the 3 spatial coordinates per atom to which the PBC have *not* been applied (for diffusion)
	v_hat = None # matrix to hold intermediate velocities during simulation (specific to the Verlet algorithm)

	# variable to save the position of all particles at the start of the equilibrium
	# used to calculate diffusion
	rEquilibriumStart = None
	rEquilibriumStartDelta = None
	rEquilibriumStartNoPBC = None
	vsqEquilibriumStart = None
	deltaRsq = None

	dist = None # matrix to hold the distances between all atoms at every time step
	F = None # matrix to hold the forces between all atoms at every time step

	# keep track of a distance histogram to calculate the correlation function from
	distHistBins = 200
	distHistBinSize = None
	distHist = None

	# matrices to hold potential and kinetic energies at every time step
	U = None
	K = None
	# matrix to hold the value of the virial coefficient at every time step
	virial = None

	autoCorrelation = None

	# set this variable to true to save all intermediate position, velocity and force data
	# RAM usage will explode
	keepRVFData = False

	# initialise_arrays: set the simulation parameters to the right values
	#                    and allocates the arrays that hold the simulation
	#                    results
	def initialise_arrays(self, desiredTIn, rhoIn, MIn):
		self.t = 0
		self.numSteps = int(self.simulationTime / self.h)
		self.equilibriumStart = int(self.equilibrationTime / self.h)
		self.diffusionTrackSteps = int(self.diffusionTrackTime / self.h)

		# set up simutation parameters
		self.M = MIn
		self.desiredT = desiredTIn
		self.rho = rhoIn
		self.n = 4*self.M**3
		self.sigma_v = np.sqrt(self.desiredT)
		self.L = np.power(self.n/self.rho, 1/3)
		self.distHistBinSize = np.sqrt(3)*self.L/(2*self.distHistBins)

		if self.keepRVFData:		
			# allocate arrays to keep *ALL* intermediate r, v and F data
			self.v = np.zeros((self.numSteps,self.n,3)) # matrix to hold the 3 velocity components per atom at every time step
			self.r = np.zeros((self.numSteps,self.n,3)) # matrix to hold the 3 spatial coordinates per atom at every time step
			self.rNoPBC = np.zeros((self.numSteps,self.n,3)) # matrix to hold the 3 spatial coordinates per atom to which the PBC have *not* been applied (for diffusion)
			self.v_hat = np.zeros((self.numSteps,self.n,3)) # matrix to hold intermediate velocities during simulation (specific to the Verlet algorithm)

			self.dist = np.zeros((self.numSteps,self.n,self.n)) # matrix to hold the distances between all atoms at every time step
			self.F = np.zeros((self.numSteps,self.n,3)) # matrix to hold the forces between all atoms at every time step
		else:
			# allocate only arrays to store the r, v and F data of the previous time step
			self.v = np.zeros((self.n,3)) # matrix to hold the 3 velocity components per atom at every time step
			self.r = np.zeros((self.n,3)) # matrix to hold the 3 spatial coordinates per atom at every time step
			self.rNoPBC = np.zeros((self.n,3)) # matrix to hold the 3 spatial coordinates per atom to which the PBC have *not* been applied (for diffusion)
			self.v_hat = np.zeros((self.n,3)) # matrix to hold intermediate velocities during simulation (specific to the Verlet algorithm)

			self.dist = np.zeros((self.n,self.n)) # matrix to hold the distances between all atoms at every time step
			self.F = np.zeros((self.n,3)) # matrix to hold the forces between all atoms at every time step
		
		self.deltaRsq = np.zeros((self.numSteps - self.equilibriumStart))

		# keep track of a distance histogram to calculate the correlation function from
		self.distHist = np.zeros((self.numSteps, self.distHistBins))

		# matrices to hold potential and kinetic energies at every time step
		self.U = np.zeros((self.numSteps))
		self.K = np.zeros((self.numSteps))
		# matrix to hold the value of the virial coefficient at every time step
		self.virial = np.zeros((self.numSteps))

		self.autoCorrelation = np.zeros((self.numSteps - self.equilibriumStart))

	# estimate the memory footprint of the algorithm in MB
	def estimate_memory_footprint(self):
		array_size = self.v.size + self.r.size + self.rNoPBC.size + self.v_hat.size + self.dist.size + self.F.size + self.deltaRsq.size + self.distHist.size + self.U.size + self.K.size + self.virial.size
		return array_size * 8 / 1e6

	def initialise_simulation(self, set_v = True):
		t = self.t = 0
		i = 0
		L = self.L
		M = self.M
		rInit = np.zeros((self.n, 3))
		for mx in range(0, M):
			for my in range(0, M):
				for mz in range(0, M):
					rInit[i,0] = L*mx/M; rInit[i,1] = L*my/M; rInit[i,2] = L*mz/M
					i += 1
					rInit[i,0] = L*(mx+0.5)/M; rInit[i,1] = L*(my+0.5)/M; rInit[i,2] = L*mz/M
					i += 1
					rInit[i,0] = L*(mx+0.5)/M; rInit[i,1] = L*my/M; rInit[i,2] = L*(mz+0.5)/M
					i += 1
					rInit[i,0] = L*mx/M; rInit[i,1] = L*(my+0.5)/M; rInit[i,2] = L*(mz+0.5)/M
					i += 1

		if self.keepRVFData:
			self.r[t,:,:] = rInit
			self.rNoPBC[t,:,:] = rInit
		else:
			self.r[:,:] = rInit
			self.rNoPBC[:,:] = rInit
		
		vInit = np.zeros((self.n, 3))

		if set_v and self.sigma_v > 0.0:
			# start with random Gaussian velocities
			vInit = np.random.normal(0.0, self.sigma_v, size=(self.n,3))
			# start with zero momentum and scale the variance to match the desired initial velocity
			vInit = (vInit - np.mean(vInit, axis=0)) / np.std(vInit) * self.sigma_v

		if self.keepRVFData:
			self.v[t,:,:] = vInit
		else:
			self.v[:,:] = vInit

		self.K[t] = 0.5*np.sum(vInit*vInit)

	def update_coordinates(self):
		t = self.t
		if self.keepRVFData:
			self.v_hat[t,:,:] = self.v[t,:,:] + self.h * self.F[t,:,:]/2
			self.r[t + 1,:,:] = (self.r[t,:,:] + self.h * self.v_hat[t,:,:]) % self.L
			self.rNoPBC[t + 1,:,:] = (self.rNoPBC[t,:,:] + self.h * self.v_hat[t,:,:])
		else:
			self.v_hat = self.v + self.h * self.F/2
			self.r += (self.h * self.v_hat)
			self.r %= self.L
			self.rNoPBC += self.h * self.v_hat
		
		# calculate all the relevant quantities for the simulation
		t += 1
		self.t += 1

		self.calculate_quantities()

		if self.keepRVFData:		
			self.v[t,:,:] = self.v_hat[t-1,:,:] + self.h*self.F[t,:,:] / 2
			self.K[t] = 0.5*np.sum(self.v[t,:,:]*self.v[t,:,:])
		else:
			self.v = self.v_hat + self.h * self.F/2
			self.K[t] = 0.5*np.sum(self.v * self.v)

		# calculate diffusion distance since the start of the equilibrium
		tSinceEq = t - self.equilibriumStart

		if tSinceEq >= 0:
			if tSinceEq % self.diffusionTrackSteps == 0: # reset the diffusion distance tracking after a set number of steps
				if self.keepRVFData:
					self.rEquilibriumStartNoPBC = np.copy(self.rNoPBC[t,:,:])
					self.rEquilibriumStart = np.copy(self.r[t,:,:])
					self.vsqEquilibriumStart = np.mean(np.sum(self.v[t,:,:]**2, axis=1))
				else:
					self.rEquilibriumStartNoPBC = np.copy(self.rNoPBC)
					self.rEquilibriumStart = np.copy(self.r)
					self.vsqEquilibriumStart = np.mean(np.sum(self.v**2, axis=1))
				self.rEquilibriumStartDelta = self.rEquilibriumStart - np.mean(self.rEquilibriumStart, axis = 0)
			curRNoPBC = None
			curR = None
			if self.keepRVFData:
				curR = self.r[t,:,:]
				curRNoPBC = self.rNoPBC[t,:,:]
			else:
				curR = self.r
				curRNoPBC = self.rNoPBC
				
			self.deltaRsq[tSinceEq] = np.mean(np.sum((curRNoPBC - self.rEquilibriumStartNoPBC)**2, axis=1))
			self.autoCorrelation[tSinceEq] = 3*np.mean((curR - np.mean(curR, axis = 0)) * self.rEquilibriumStartDelta)

	def calculate_quantities(self):
		# we can't use instance variables inside a jitted function, so we provide this little wrapper function
		t = self.t
		if self.keepRVFData:
			self.dist[t,:,:], self.F[t,:,:], self.distHist[t,:], self.U[t], self.virial[t] = calculate_quantities_jit(self.r[t,:,:], self.L, self.distHistBins, self.distHistBinSize)
		else:
			self.dist, self.F, self.distHist[t,:], self.U[t], self.virial[t] = calculate_quantities_jit(self.r, self.L, self.distHistBins, self.distHistBinSize)

	def scale_to_temperature(self):
		# average the velocities over the second half of the last rescaling period
		halfRP = int(self.rescalingPeriod/2)
		lbda = np.sqrt((self.n - 1)*3*self.desiredT*halfRP/(2*np.sum(self.K[self.t-halfRP:self.t])))
		
		if self.keepRVFData:
			self.v[self.t,:,:] *= lbda
		else:
			self.v *= lbda

	def do_simulation(self):

		np.random.seed()
		self.t = 0
		self.initialise_simulation()
		self.calculate_quantities()

		while(self.t < self.numSteps - 1):
			if (self.t*self.h < self.rescalingTime and self.t > 0 and self.t % self.rescalingPeriod == 0):
				self.scale_to_temperature()
			if (self.t % int(self.numSteps*self.reportFreq) == 0):
				print_fl("Simulation progress: " + str(int(self.t*100/self.numSteps)) + "%")  
			self.update_coordinates()
		print_fl("Simulation finished")

	def test_update_coordinates(self):
		self.t = 0
		self.update_coordinates()
		self.t = 0

	# post-simulation methods to extract results from the simulation data

	def result_diffusion_distance(self):
		return self.deltaRsq

	def result_diffusion_ballistic_velocitysq(self):
		return self.vsqEquilibriumStart

	def result_kinetic_energy(self):
		return self.K[self.equilibriumStart:]

	def result_potential_energy(self):
		return self.U[self.equilibriumStart:]

	def result_virial(self):
		return self.virial[self.equilibriumStart:]

	# this temperature calculation is slightly off (N should be replaced by N-1)
	# this is fixed in transcribe_results, along with the temperature dependant pressure
	def result_temperature(self):
		return 2*self.result_kinetic_energy()/(3*self.n)

	def result_pressure(self):
		return 1 + 1/(3 * self.n * np.mean(self.result_temperature())) * np.mean(self.virial[self.equilibriumStart:])

	def result_distance_histogram(self):
		return self.distHist[self.equilibriumStart:,:]

	def result_correlation_function(self):
		r_g = np.linspace(0.5*self.distHistBinSize, (self.distHistBins - 0.5)*self.distHistBinSize, self.distHistBins)
		g = np.mean(self.distHist[self.equilibriumStart:,:], axis=0)/(4*np.pi*self.distHistBinSize*r_g**2)*2*self.L**3/(self.n*(self.n-1))
		return r_g, g

	def result_cv(self):
		delta_K_sq = np.var(self.K[self.equilibriumStart:])
		K_sq = np.mean(self.K[self.equilibriumStart:])**2

		C_v = 3 * K_sq / (2*K_sq - 3*self.n*delta_K_sq)
		return C_v

	def result_autocorrelation(self):
		return self.autoCorrelation

	# save all the data generated by result_-functions
	def save_results(self, filename):
		result_methods = [rm for rm in dir(self) if rm.startswith("result_")]
		results = dict()
		for rm in result_methods:
			results[rm[7:]] = getattr(self, rm)()

		results["rho"] = self.rho
		results["desiredT"] = self.desiredT
		results["M"] = self.M
		results["n"] = self.n
		results["simulationTime"] = self.simulationTime
		results["equilibrationTime"] = self.equilibrationTime
		results["h"] = self.h
		results["distHistBinSize"] = self.distHistBinSize
		results["distHistBins"] = self.distHistBins

		np.savez(filename, **results)


	

