################################################################""
#
#   var_standard.py

#   (c) Alex AYET - 02/2017

#  Parameters list used in analytical_CTW.py
#  
#  This a setup to compare with Killpatrick's 15M/S simulations

#####################################################################

import numpy as np

#--- Directory where the results are saved
RESDIR = '../data/'

#------ Numerical param.
#--Number of horizontal points
N_x = 35
dx = 100000.
#--Number of vertical points
N_z = 100

#------- Physical

#--Coriolis (Latitude North)
phi = 42.
#/Do not change
rad = phi*np.pi/180.
omega = 7.27e-5
f = 2*omega*np.sin(rad) 
#/end

#--Gravity
g = 9.8

#--Ref. temp.
T_0 = 280.

#-- SST field

gap = 3.
deltal= 2. #in number of points

SST = np.fromfunction(lambda i: gap*0.5*(1+np.tanh((i-28)/4)), (N_x+1,))


#--Geostrophic wind:
Ug = 15.
#--MABL max and min height:
HMAX = 900.
HMIN = 500. 
#--Diffusion coefficient parameters: 
K_00 = 0.0001#.0001 
K_01 =  0.

k_0 = 0.0001
k_1 = 0.1 
k_2 = 10.
k_3 = 10.

#--- COMPUTATION OF THE RELATED QUANTITIES

#-MABL height and its derivative w.r.t. theta
delta = (((HMAX-HMIN)/gap)*SST[1:] + HMIN) 
Ddelta = ((HMAX-HMIN)/gap)


#--- Diffusion coefficients
K_1 = np.zeros(N_x)
for i in xrange(N_x):
	K_1[i] = k_0#*((k_1)**SST[i])

Kmax = np.zeros(N_x)
for i in xrange(N_x):
	Kmax[i] = k_2 + k_3*SST[i]

K_0 = np.zeros(N_x)
for i in xrange(N_x):
	K_0[i] = K_00#+ K_01*SST[i]

A = Kmax
B = (K_1 - K_0)/delta
C = 2*(K_0 + K_1 - 2*Kmax)/(delta**2)

#################################
#----- Horizonal discretization (USED IN plot.py ONLY)
DZ = np.max(delta)/(N_z -1) 
#################################
