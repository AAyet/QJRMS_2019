import sys
import mpmath as mp
import numpy as np
import matplotlib.pyplot as plt
import time
import logging

mp.pretty = True


###############################################################################
#
#   analytical_CTW.py

#   (c) Alex AYET - 06/2019


#    Computation of an analytical solution for the SST front wind problem
#    If using interactive python, recommended to run "%reset -f" before each run

#    It generates 
#    -wind(Ug,gap).py which saves the total solution (total wind) (u + iv)
#    -nondim(Ug,gap).py: the adimentional parameters.
#    -a logging file with various informations on the run.

#################################################################################

#------ INPUT VARIABLES
#--- change variables file to change the parameters

#import var_15 as var
import var_5 as var

#-----------------------



#----------- DEFINITION OF LEGENDRE FUNCTIONS (do not change) -----
#-------------------------------------------------

#First kind
def LegP(lamb,z):
	
	return mp.hyp2f1(lamb+1, -lamb,1,0.5 - 0.5*z)	

#Second kind:

def LegQ(lamb,z):
	if z.imag >= 0:
		exa = mp.exp(1j*0.5*mp.pi*(-lamb-1))
		exb = mp.exp(1j*0.5*mp.pi*(-lamb))
	else:
		exa = mp.exp(-1j*0.5*mp.pi*(-lamb-1))
		exb = mp.exp(-1j*0.5*mp.pi*(-lamb))
	if z ==1 : 
		return 1.
	else:
		a = (mp.fac(-0.5+0.5*lamb)/(2*mp.fac(0.5*lamb)))*exa*mp.hyp2f1(-0.5*lamb, 0.5+ 0.5*lamb, 0.5, z**2)
		b = z*(mp.fac(0.5*lamb)/mp.fac(-0.5+0.5*lamb))*exb*mp.hyp2f1(0.5-0.5*lamb,1+0.5*lamb, 1.5, z**2)
	
		return complex((np.pi)**(.5)*(a+b))

#-----------------------------------------
#--------------PARAMETERS (from imported file)
#-----------------------------------------

#--Saving directory
RESDIR= var.RESDIR

#------ Numerical
#--Number of horizontal points
N_x = var.N_x
dx = var.dx
#--Number of vertical points
N_z = var.N_z

#------- Physical
#--Coriolis
f = var.f

#--Gravity:
g = var.g
#--Ref. temp.:
T_0 = var.T_0

#SST, and SST gap and typical length scale
gap = var.gap
deltal= var.deltal
SST = var.SST

#--Geostrophic wind:
Ug = var.Ug
#--MABL max and min height:
HMAX = var.HMAX 
HMIN = var.HMIN
#--Diffusion coefficient parameters: 
K_00 = var.K_00
K_01 = var.K_01

k_0 = var.k_0
k_1 = var.k_1
k_2 = var.k_2
k_3 = var.k_3

alpha = 1.

#-------------------------------------------
#-------------END PARAMETRES ---------------
#-------------------------------------------
#---Computes Rossby number (for logging only):
rossby = np.real(Ug)/(f*deltal*dx)

#----- LOGGING BEGINS:
logging.basicConfig(filename=RESDIR+"CTW.log", level=logging.INFO)

ti= time.strftime('%d/%m %H:%M', time.localtime())
logging.info("START: geos wind %s, gap %s at %s"%(Ug, gap,ti))
logging.info("rossby number: %s"%(rossby))
start_time = time.clock()

#-----------ADDITIONAL PARAMETERS:
#----Diffusion coefficient parametrization:
K_1 = var.K_1
Kmax = var.Kmax
K_0 = var.K_0


#----MABL height
delta = var.delta
#--- a is half the total BL height (used for convenience):
a = 0.5*delta

#--- SST gradient:
DT = (SST[1:] - SST[:-1])/dx

#-----derivative of delta w.r.t. theta
Ddelta = var.Ddelta

#----- Horizonal discretization
dz = np.max(delta)/(N_z -1)      
t = (delta/dz) +1

#--- Number of points in the MABL
N = np.floor(t).astype(int) 

#--- Other parametrization (see appendix): C*Z**2 + B*Z + A = K(z)

A = Kmax
B = (K_1 - K_0)/delta
C = 2*(K_0 + K_1 - 2*Kmax)/(delta**2)

#--- Logging of the different nondim parameters (maximum value over the domain)
Ekk = (B**2 - 4*A*C)/(2*C*f*delta**2)
Ekman = np.zeros((N_z,N_x))
for i in xrange(N_z):
	for j in xrange(N_x):
		Ekman[i,j] = ((i+1)*dz)**2/(2*a[j])**2
logging.info("Ekmann number: %s"%(np.max(Ekman)))

PC = (g*gap*delta)/(T_0*f*np.real(Ug)*deltal*dx) 
logging.info("PC non dim number: %s"%(np.max(PC)))



#-------FUNCTIONS AND ARGUMENTS
#---- Returns the physical position along x and z respectiely
def PX(i):
	return (-(N_x/2.)+ i)*dx

def PZ(i):
	return dz*i
#---- Legendre functions order
lamb = 0.5*(np.sqrt(1+(4/C)*1j*f) -1) 
# --- Argument of the Legendre functions (i: z and j: x):  can be changed with a minus sign
#     (equivalent to changing i to 1/i in the change of variables)
def ch(i,j):
	return (2*C[j]*(PZ(i) - a[j])+ B[j])/(np.sqrt(B[j]**2-4*A[j]*C[j]))

#---- Argument of the Leg. functions at the boundaries
down = 	(-(2*C*a)+ B)/(np.sqrt(B**2-4*A*C))
up = ((2*C*a)+ B)/(np.sqrt(B**2-4*A*C))


#----- SOLUTION TERMS (j: z and i: x):

#-- RHS: particular solution
P_0 = 1e5
R = 208.
def RHS(j,i): 
	return (g/(T_0))*DT[i]*((C[i]*(2*a[i] + SST[i]*Ddelta)-B[i])/(1j*f*(-2*C[i]+1j*f))- PZ(j)/(-2*C[i] + 1j*f)+ (2*a[i]+SST[i]*Ddelta)/(1j*f)) 

#---- Stratification functions
def D(i):
	return 1./(LegP(lamb[i], up[i])*LegQ(lamb[i], down[i]) - LegP(lamb[i], down[i])*LegQ(lamb[i], up[i]))

# -- Function H
#    p is the lower index of H (0 or delta <-> 0 or 1)
def H(j,i,p):
	if p == 0:
		pos = up
	elif p == 1:
		pos = down
	return D(i)*(LegP(lamb[i],ch(j,i))*LegQ(lamb[i],pos[i]) - LegQ(lamb[i], ch(j,i))*LegP(lamb[i], pos[i]))

#------------------------------
#------FINAL SOLUTION

sol = np.zeros((N_z,N_x), dtype='complex')+ Ug 

part = np.zeros((N_z,N_x), dtype= 'complex')

for j in xrange(N_x):
	for i in xrange(N[j]):
		part[i,j] = RHS(i,j)
		sol[i,j] = sol[i,j] + part[i,j]  + (part[0,j] + Ug)*H(i,j,0) - part[N[j]-1,j]*H(i,j,1)  

#---- SAVE DATA

np.save(RESDIR+'wind(%s,%s)'%(int(Ug), int(gap)), sol)
np.save(RESDIR+'N(%s,%s)'%(int(Ug), int(gap)), N)


#--- END LOGGING
mi = float((time.clock() - start_time)/60.)
minu = np.floor(mi)
sec = (mi-minu)*60. 
logging.info(str(minu)+"min "+str(sec)+ "seconds")
logging.info("END")


