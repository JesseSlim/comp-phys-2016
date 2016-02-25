# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 14:58:19 2016

@author: Sybren_Zwetsloot
"""

# Clear all current variables
import sys
#sys.modules[__name__].__dict__.clear()

import math
import time
import numpy as np
import random
from numba import jit
import matplotlib.pyplot as plt

# Functions
#@jit

# INIT
N = 2                       # Integer value
n = math.pow(N,3)*4         # Molecule count
n = int(n)
dens = 0.8                  # dens = N / V
V = n / dens                # Volume density
D = math.pow(V,1/3)         # Size of the box

pos     = np.zeros([n,3])   # Array with positon
vel     = pos               # Array with velocities
force   = pos               # Array with forces

c = 0
for i in range(0,N):
    for j in range(0,N):
        for k in range(0,N):
            pos[c,:] = [(i+0.5)*D,(j+0.5)*D,(k+0.5)*D]
            pos[c+1,:] = [(i+1)*D,(j+1)*D,(k+0.5)*D]
            pos[c+2,:] = [(i+1)*D,(j+0.5)*D,(k+1)*D]
            pos[c+3,:] = [(i+0.5)*D,(j+1)*D,(k+1)*D]
            c = c + 4
            
for c in range(0,n):
    vel[c,:] = np.random.rand(1,3)

t = 0
dt = 0.001
CalculateForces()

#@jit
def CalculateForces() :
    global force
    for i in range(0,n):
        for j in range(0,n):
            dx = pos[i,0] - pos[j,0]
            dy = pos[i,1] - pos[j,1]
            dz = pos[i,2] - pos[j,2]
            dx = dx - round(dx / (D)) * D
            dy = dy - round(dy / (D)) * D  
            dz = dz - round(dz / (D)) * D  
            d = math.sqrt(dx**2+dy**2+dz**2)
            F = 12*(2*d**(-13) - d**(-7))
            force[i,0] += F * dx / d
            force[i,1] += F * dy / d
            force[i,2] += F * dz / d
    return