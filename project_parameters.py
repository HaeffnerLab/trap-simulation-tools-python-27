"""This script contains the parameters for developing the project trapping field.
Lower order functions sould rely entirely on data passed to them as configuration attributes."""

import numpy as np
import scipy.io as io
import datetime

# International System of Units
qe=1.60217646e-19
me=9.10938188e-31
mp=1.67262158e-27

# Universal Parameters
save = True                  # save the converted data to a python pickle 
debug =   1               # control if sanity checking plots are made; same as plottingOption

#################################################################################
################################# For Import ####################################
#################################################################################
"""Includes project parameters relevant to import_data to build entire project in one script. Previously systemInformation."""
projectName = 'project_test_code_eurotrap' # Arbitrary
simulationDirectory='C:\\Python27\\trap_simulation_software\\eurotrap-gebhard\\'
baseDataName = 'eurotrap-pt'# Excludes the number at the end to refer to a set of simulations
timeNow = datetime.datetime.now().date()
fileName = projectName+'_'+str(timeNow) # For saving the data structure. Old and optional.
startingSimulation = 1      # nStart: index of the trap simulation file on which you want to start importing data
numSimulations = 6          # old nMatTot
dataPointsPerAxis = 21      # old NUM_AXIS 
nonGroundElectrodes = 22    # old NUM_ELECTRODES 
numUsedElectrodes = 22      # old NUM_USED_ELECTRODES
savePath = 'C:\\Python27\\trap_simulation_software\\data\\' # directory to save data at
perm = [1,2,0] # specific to BEM-solver text files; BEM is typically in order z,x,y, so we want perm [1,2,0], not original [2,1,0]
###COORDINATES Nikos code uses y- height, z - axial, x - radial
#if drawing uses x - axial, y - radial, z - height, use perm = [1,2,0] (Euro trap)
#if drawing uses y - axial, x - radial, z - height, use perm = [0,2,1] (Sqip D trap, GG trap)
#if drawing uses Nikos convention, use perm = [0,1,2]


""" Here define the electrode combinations. 
The convention is physical electrode -> functional electrode.
If electrodes 1 and 2 are combined into one electrode, then enter [1 1; 2 1; 3 2;...
If electrodes 1 and 4 are not in use (grounded), then enter [1 0; 2 1; 3 2; 4 0...
nomGroundElectrodes (i.e. last) is the RF electrode.
nomGroundElectrodes-1 (i.e. before the RF) is the center electrode.
electrodeMapping determines the pairing. 
manualElectrodes determines the electrodes which are under manual voltage control. 
It has NUM_ELECTRODES elements (i.e. they are not connected to an arbitrary voltage, not to multipole knobs).
All entries != 0 are under manual control, and entries = 0 are not under manual control."""  
electrodeMapping = np.array([[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7],[8,8],[9,9],[10,10],
                    [11,11],[12,12],[13,13],[14,14],[15,15],[16,16],[17,17],[18,18],
                    [19,19],[20,20],[21,21],[22,22]])                                                
dcVoltages =       [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1] # VMULT for expand_field before setdc can be run                
manualElectrodes = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] # VMAN for dcpotential_instance in expand_field
weightElectrodes = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] # IMAN for dcpotential_instance in expand_field
usedMultipoles = [1,1,1,1,1,1,1,1]

#################################################################################
################################# For Field #####################################
#################################################################################
"""fieldConfig, previously trapConfiguration; not all variables will be passed to output
Parameters used for get_trapping_field, expand_field, and trap_knobs
Some of the required parameters are listed with import config."""
#pathName = savePath+fileName+'_simulation_'
pathName = savePath+projectName+'_simulation_'
position = -555 # trapping position along the trap axis (microns)
zMin = -630 # lowest value along the rectangular axis
zMax = -510 # highest value along the rectangular axis
zStep = 20  # range of each simulation
name = 'tk_test'    # name to save trapping field as
rfBias = False                 

# expand_field 
trap = savePath+name+'.pkl'
Xcorrection = 0
Ycorrection = 0
L = 2
NUM_DC = 22
NUM_Center = 10
E = [0,0,0] # Old Ex,Ey,Ez

# trap_knobs
reg=True

#################################################################################
################################# From ppt2  ####################################
#################################################################################
#instance config
driveAmplitude = 100     # Applied RF amplitude for ppt3 analysis
driveFrequency = 40e6    # RF frequency for ppt3 analysis
findEfield       = False # determine the stray electric field for given dc voltages
justAnalyzeTrap  = True  # do not optimize, just analyze the trap, assuming everything is ok
rfplot = '1D plots' # dimensions to plot RF with plotpot
dcplot = '1D plots' # dimensions to plot DC with plotpot