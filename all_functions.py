"""This is all functions and scripts used by the simulation. All relevant abstraction in project_parameters and analyze_trap."""

# Functions to absorb as helpers?: 
# d_e to ppt3
# meshslice to plotpot
# spherharmcmp to spherharmq
# plotN to trap_knobs
# sumofefield to exactsaddle
# p2d and trapdepth to pfit
# pfit to ppt3

# Primary Functions (helpers contained within them)
def import_data():
    """Originally importd by Mike, modified by Gebhard Oct 2010. 
    Redefined conventions, and cleaned up and combined with later
    developments by Nikos Jun 2013. Converted to Python by William Jan 2014.
    
    Imports BEM-solver data according to Gebhard's file saving notation.
    If there is E field data, then Gebhart's 2010 if loop imports it as well.
    Takes .txt file as input and converts to python pickles.
    The potentials for the trap electrodes and the grid vectors
    
    importd.py can import multiple .txt files (see the loop).
    The number of files can be adjusted by nMatTot.
    
    All the conventions concerning which electrodes are used and which are bound together, 
    defined in project_parameters, are implemented here.
    
    The DC electrodes are grouped in two categories: EL_DC (multipole controlled), and mEL_DC (manual voltage controlled)  
    hard-codes to reorganize struct for specific electrode configuration
    
    Rules: 
    * All the electrodes are initially assumed to be DC.
    * The sequence for counting DC electrodes runs through the left side of the RF (bottom to top), right side of
    the RF (bottom to top), center electrodes inside of the RF (left center, then right center), and finally RF.
    * To specify that an electrode is grounded, go to project_parameters and set the corresponding parameter in
    electrodeMapping to 0."""
    
    from project_parameters import dataPointsPerAxis,nonGroundElectrodes,numUsedElectrodes,electrodeMapping,manualElectrodes,save,debug
    from project_parameters import projectName,baseDataName,simulationDirectory,fileName,startingSimulation,numSimulations,savePath,perm
    from treedict import TreeDict
    import pickle
    import numpy as np

    # renaming for convenience
    na=dataPointsPerAxis 
    ne=nonGroundElectrodes 
    nue=numUsedElectrodes 
    em = electrodeMapping
    me = manualElectrodes
    
    # iterate through each simulation text file
    for iterationNumber in range(startingSimulation,numSimulations+1):
        #########################################################################################
        # Part 0: Check if data already exists 
        def fileCheck(iterationNumber):
            """Helper function to determine if there already exists imported data."""
            try:
                #file = open(savePath+fileName+'_simulation_{}'.format(iterationNumber)+'.pkl','rb')
                file = open(savePath+projectName+'_simulation_{}'.format(iterationNumber)+'.pkl','rb')
                file.close()
                if iterationNumber==numSimulations:
                    return 'done'
                return fileCheck(iterationNumber+1)
            except IOError: 
                print ('No pre-imported data in directory for simulation {}.'.format(iterationNumber))
                return iterationNumber
        iterationNumber=fileCheck(iterationNumber) # lowest iteration number that is not yet imported
        if iterationNumber=='done':
            return 'All files have been imported.'
        
        #########################################################################################
        # PART I: Read txt file
        print('Importing '+''+baseDataName+str(iterationNumber)+'...')
        dataName=(simulationDirectory+baseDataName+str(iterationNumber)+'.txt')
        
        #1) check if there is BEM-solver data to import
        try: 
            DataFromTxt=np.loadtxt(dataName,delimiter=',') 
        except IOError:
            print ('No BEM-solver data to import for simulation {}. Import complete.'.format(iterationNumber))
            import sys
            sys.exit()
            
        #2) build the X,Y,Z grids
        X = [0]
        Y = [0]
        Z = DataFromTxt[0:na,2]
        for i in range(0,(na)):
            if i==0:
                X[0]=(DataFromTxt[na**2*i+1,0])
                Y[0]=(DataFromTxt[na*i+1,1])
            else:
                X.append(DataFromTxt[na**2*i+1,0])
                Y.append(DataFromTxt[na*i+1,1])
        X = np.array(X).T
        Y = np.array(Y).T
        XY = np.vstack((X,Y))
        coord=np.vstack((XY,Z))
        coord=coord.T
        X = coord[:,perm[0]]
        Y = coord[:,perm[1]]
        Z = coord[:,perm[2]]
        
        #3) load all the voltages and E vector into struct using dynamic naming 
        struct=TreeDict() # begin intermediate shorthand.
        for el in range(ne+1): #el refers to the electrode.
            struct['phi_{}'.format(el)]=np.zeros((na,na,na))
            struct['Ex_{}'.format(el)]=np.zeros((na,na,na))
            struct['Ey_{}'.format(el)]=np.zeros((na,na,na))
            struct['Ez_{}'.format(el)]=np.zeros((na,na,na))
            for i in range(na):
                for j in range (na):
                    lb = na**3*el + na**2*i + na*j #lower bound
                    ub = na**3*el + na**2*i + na*j + na #upper bound
                    struct['phi_{}'.format(el)][i,j,:]=DataFromTxt[lb:ub,3]
                    ## if loop by Gebhard, Oct 2010; used if there is E field data in BEM
                    if (DataFromTxt.shape[1]>4): ### i.e. Ex,Ey,Ez are calculated in bemsolver (old version), fast
                        struct['Ex_{}'.format(el)][i,j,:]=DataFromTxt[lb:ub,4]
                        struct['Ey_{}'.format(el)][i,j,:]=DataFromTxt[lb:ub,5]
                        struct['Ez_{}'.format(el)][i,j,:]=DataFromTxt[lb:ub,6]
                    else:
                        ## i.e. Ex, Ey, Ez are NOT calculated in bemsolver (slow bemsolver, more exact).
                        ## Erf will be calculated by the numerical gradient in ppt2.m
                        struct['Ex_{}'.format(el)][i,j,:]=0
                        struct['Ey_{}'.format(el)][i,j,:]=0
                        struct['Ez_{}'.format(el)][i,j,:]=0
            struct['phi_{}'.format(el)]=np.transpose(struct['phi_{}'.format(el)],perm)
            struct['Ex_{}'.format(el)]=np.transpose(struct['Ex_{}'.format(el)],perm)
            struct['Ey_{}'.format(el)]=np.transpose(struct['Ey_{}'.format(el)],perm)
            struct['Ez_{}'.format(el)]=np.transpose(struct['Ez_{}'.format(el)],perm)
        del DataFromTxt
        
        #########################################################################################
        # PART II: Organize the electrodes in data according to the trap configurtation.
        sim=TreeDict()
        sim.X,sim.Y,sim.Z=X,Y,Z                # set grid vectors
        sim.EL_RF = struct['phi_{}'.format(0)] # set RF potential field
        
        #1) initialize NUM_USED_DC electrodes
        for iii in range(1,nue+1):
            sim['EL_DC_{}'.format(iii)],sim['mEL_DC_{}'.format(iii)]=np.zeros((na,na,na)),np.zeros((na,na,na))
        if (em[em.shape[0]-1,0]!=ne) or (em[em.shape[0]-1,1]!=nue):
            print('import_data: There seems to be a problem with your mapping definition. Check electrodeMapping.')
            
        #2) add each electrode to the combination where it belongs 
        # If electrodeMapping entry is 0, then the electrode is not used. The last electrode is the RF, and it is added by hand.
        for iii in range(ne-1):
            if (float(em[iii,1]/abs(em[iii,1]))):
                sim['EL_DC_{}'.format(em[iii,1])] += struct['phi_{}'.format(em[iii,1])]
            elif manualElectrodes[iii]:
                sim['mEL_DC_{}'.format(iii)] += struct['phi_{}'.format(iii)]
        if (em[ne-1,1]/abs(em[ne-1,1])):
            sim['EL_DC_{}'.format(em[ne-1,0])] = sim.EL_RF
        elif me[ne-1]:
            sim['mEL_DC_{}'.format(ne-1)] = sim.EL_RF
    
        if debug: # Plot the RF potential
            Ef = sim['EL_DC_{}'.format(em[iii+1,1])]  
            E=np.zeros((na,na))
            from all_functions import plotpot
            import matplotlib.pyplot as plt
            from matplotlib import cm
            import mpl_toolkits.mplot3d.axes3d as p3
            print(plotpot(Ef,X,Y,Z,'1D plots','Debug: RF Plot sim{}'.format(iterationNumber)))
            for a in range(na):
                for b in range(na):
                     E[a,b] = Ef[a,b,na-1]
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            x,y=np.meshgrid(sim.X,sim.Y)
            surf = ax.plot_surface(x,y,E,cmap=cm.coolwarm,antialiased=False)
            plt.title('Debug: Plotting the RF potential sim{}'.format(iterationNumber))
            plt.show() 
        
        #3) save the particular simulation as a pickle data structure
        if save == True:
            #name=savePath+fileName+'_simulation_{}'.format(iterationNumber)+'.pkl'
            name=savePath+projectName+'_simulation_{}'.format(iterationNumber)+'.pkl'
            print ('Saving '+name+' as a data structure...')
            output = open(name,'wb')
            pickle.dump(sim,output)
            output.close()
            
    return 'Import Complete'
    
def get_trap():
    """Originally getthedata.
    Create a new "trap" structure centered around the given position and composed of portions of the adjacent simulations.
    The electrodes are ordered as E[1],...,E[NUM_DC]=E[RF].
    The NUM_DC-1 is the center electrode bias, and the NUM_DC is the RF electrode bias.
    (If center and RF are used.)
    Connect together a line of cubic matrices to describe a rectangular prism of data.
    The consecutive data structures must have overlaping first and last points. 
    
    Also creates field configuration attributes on the trap that will be used by lower order functions.
    Recall that the grid vectors X,Y,Z are still attributes of potentials now and will become attributes of instance later.
    Nikos 2009.
    Cleaned 26-05-2013, 10-23-2013
    Converted to Python by William Jan 2014"""
    
    #0) define parameters
    from project_parameters import pathName,savePath,position,zMin,zMax,zStep,save,debug,name,rfBias,electrodeMapping,dataPointsPerAxis,timeNow
    from project_parameters import dcVoltages,manualElectrodes,weightElectrodes,numUsedElectrodes,usedMultipoles,numSimulations,qe,mp
    import pickle, pprint
    import numpy as np
    from treedict import TreeDict
    tf=TreeDict() # begin shorthand for trap data structure
    nue=numUsedElectrodes

    #1) Check if the number of overlapping data structures is teh same as the number of simulations.
    numSim=int((zMax-zMin)/zStep)
    if numSim!=numSimulations:
        raise Exception('Inconsistence in simulation number. Check project_parameters for consistency.')
    
    #2) Define a background for files. 
    zLim=np.arange(zMin,zMax,zStep) 
    
    #3) helper function to find z-position of ion
    def find_index(list,position): # Find which simulation position is in based on z-axis values.
        """Finds index of first element of list higher than a given position. Lowest index is 1, not 0"""
        # replaces Matlab Code: I=find(zLim>position,1,'first')-1
        i=0
        for elem in list:
            if elem>position:
                index=i
                return index
            else: 
                index=0
            i += 1
        return index 

    index=find_index(zLim,position)
    if (index<1) or (index>numSimulations):
        raise Exception('Invalid ion position. Quitting.')

    #4) determine which side of the simulation the ion is on
    pre_sign=2*position-zLim[index-1]-zLim[index] # determines 
    if pre_sign==0:
        # position is exactly halfway between
        sign=-1 # could be 1 as well
    else:
        sign=int(pre_sign/abs(pre_sign))
        
    #5) If position is in the first or last grid, just use that grid.
    if (index==1) and (sign==-1): 
        print pathName+'1.pkl'
        file = open(pathName+'1.pkl','rb')
        tf.potentials = pickle.load(file)
        #pprint.pprint(tf.potentials)
        file.close()
    
    #6) If the ion is in the second half of the last grid, just use the last grid. 
    elif (index==numSimulations) and (sign==1): 
        file = open(pathName+'{}.pkl'.format(numSimulations),'rb')
        tf.potentials = pickle.load(file)
        #pprint.pprint(tf.potentials)
        file.close()
    
    #7) Somewhere in between. Build a new set of electrode potentials centered on the position.
    else:
        #a) open data structure
        file = open(pathName+'{}.pkl'.format(index),'rb')
        tf.potentials = pickle.load(file)
        #pprint.pprint(tf.potentials)
        file.close()
        lower=position-zStep/2 # lower bound z value 
        upper=position+zStep/2 # upper bound z value
    
        shift=int(pre_sign)    # index to start from in left sim and end on in right sim
        if shift < 0:
            index -= 1
            shift = abs(shift)
        
        #b) open left sim
        file1 = open(pathName+'{}.pkl'.format(index),'rb')
        left = pickle.load(file1)
        #pprint.pprint(tf.potentials)
        file1.close()
        
        #c) open right sim
        file2 = open(pathName+'{}.pkl'.format(index+1),'rb')
        right = pickle.load(file2)
        #pprint.pprint(tf.potentials)
        file.close()
        
        #d) build bases
        cube=tf.potentials.EL_DC_1 # arbitrary electrode; assume each is cube of same length
        w=len(cube[0,0,:])         # number of elements in each cube; width 
        potential=tf.potentials    # base potential to write over
        Z=potential.Z              # arbitrary axis with correct length to plot against
        
        #e) build up trap
        for el in range(nue):
            el+=1
            m,n=w-shift,0
            temp=np.zeros((w,w,w)) #placeholder that becomes each new electrode
            left_el=left['EL_DC_{}'.format(el)]
            right_el=right['EL_DC_{}'.format(el)]
            for z in range(shift-1,w-1):
                temp[:,:,n]=left_el[:,:,z]
                n+=1
            for z in range(shift): #ub to include final point
                temp[:,:,m]=right_el[:,:,z]
                m+=1
            potential['EL_DC_{}'.format(el)]=temp
        
    #8) check if field generated successfully
    if debug:
        import matplotlib.pyplot as plt
        plt.plot(Z,np.arange(len(Z)))
        plt.title('get_trap: contnuous straight line if successful')
        plt.show()  
    #9) assign configuration variables to trap; originally trapConfiguration
    trap = tf
    c=trap.configuration
    c.rfBias=rfBias
    c.charge=qe
    c.mass=mp
    c.numUsedElectrodes=numUsedElectrodes
    c.electrodeMapping = electrodeMapping                        
    c.manualElectrodes = manualElectrodes
    c.usedMultipoles = usedMultipoles 
    c.position = position 
    c.dataPointsPerAxis = dataPointsPerAxis
    c.date = timeNow # consider moving to systemInformation rather than configuration?
    trap.configuration=c

    if save:
        import pickle
        name=savePath+name+'.pkl'
        print ('Saving '+name+' as a data structure...')
        output = open(name,'wb')
        pickle.dump(trap,output)
        output.close()
    
    return 'Constructed trap.'

def expand_field():
    """Old regenthedata
    Regenerates the potential data for all electrodes using multipole expansion to order L.
    Also returns a field of trap, configuration.multipoleCoefficients, which contains the multipole coefficients for all electrodes.
    
    The electrodes are ordered as E(1), ..., E(NUM_DC)=E(RF)
    i.e. the NUM_DC-1 is the center electrode bias, and the NUM_DC is the RF electrode bias
    (if center and RF are used)
          ( multipoles    electrodes ->       )
          (     |                             )
    M =   (     V                             )
          (                                   )
    Multipole coefficients only up to order 8 are kept, but the coefficients are calculated up to order L.

    trap is the path to a data structure that contains an instance with the following properties
    .DC is a 3D matrix containing an electric potential and must solve Laplace's equation
    .x,.y,.z are the vectors that define the grid in three directions
    
    Xcorrection, Ycorrection: optional correction offsets from the RF saddle pointm,
                              in case that was wrong by some known offset
    position: the axial position where the ion sits
    L: order of the multipole expansion
    NUM_DC and NUM_Center: number of DC electrodes and number of center electrodes

    Nikos June 2009
    Cleaned up 26-05-2013, 10-23-2013

    The correction Xcorrection,Ycorrection are parameters allowing one to offset the RF saddle point,
    for example to fix a wrong RF simulation.
    
    William Python Jan 2014"""
    
    #0) establish parameters
    from project_parameters import trap,Xcorrection,Ycorrection,L,NUM_DC,NUM_Center,save,debug,name
    print save,debug
    from project_parameters import dcVoltages,manualElectrodes,weightElectrodes,E
    from all_functions import spherharmxp,spherharmcmp,spherharmq,findsaddle,exactsaddle,plotpot,dcpotential_instance
    import numpy as np
    import pickle, pprint
    file = open(trap,'rb')
    print trap,file
    tf = pickle.load(file)
    #pprint.pprint(tf)
    file.close()
    if tf.configuration.expand_field==True:
        return 'Field is already expanded.'
    if tf.instance.check!=True:
        n=tf.configuration.dataPointsPerAxis
        VMULT= dcVoltages       #all 1
        VMAN = manualElectrodes #all 0
        IMAN = weightElectrodes #all 0
        # run dcpotential_instance to create instance configuration
        dcpotential_instance(trap,VMULT,VMAN,IMAN,E,True)
    # open updated trap
    file = open(trap,'rb')
    tf = pickle.load(file)
    #pprint.pprint(tf)
    file.close()
    V=tf.instance.DC 
    X=tf.instance.X
    Y=tf.instance.Y
    Z=tf.instance.Z
    origin=findsaddle(V,X,Y,Z,3)
    print tf.instance.check!=True
    if debug:
        plotpot(V,X,Y,Z,'1D plots','V.DC',origin)
    
    tc=tf.configuration #intermediate configuration
    position = tc.position
    tc.EL_RF = tf.potentials.EL_RF
    if Xcorrection:
        print('expand_field: Correction of XRF: {} mm.'.format(str(Xcorrection)))
    if Ycorrection:
        print('expand_field: Correction of YRF: {} mm.'.format(str(Ycorrection)))
    [x,y,z] = np.meshgrid(X,Y,Z)
    # Order to expand to in spherharm for each electrode.
    order = np.zeros(NUM_DC)
    order[:]=int(L)
    N=(L+1)**2 # L is typically 2, making this 9
    
    #1) Expand the rf about the grid center, regenerate data from the expansion.
    print 'Expanding RF potential'
    Irf,Jrf,Krf = int(np.floor(X.shape[0]/2)),int(np.floor(Y.shape[0]/2)),int(np.floor(Z.shape[0]/2))
    Xrf,Yrf,Zrf = X[Irf],Y[Jrf],Z[Krf]
    if debug:
        plotpot(tc.EL_RF,X,Y,Z,'1D plots','initial EL_RF',[Irf,Jrf,Krf])
    Qrf = spherharmxp(tc.EL_RF,Xrf,Yrf,Zrf,X,Y,Z,L)
    
    print 'Comparing RF potential'
    tc.EL_RF = spherharmcmp(Qrf,Xrf,Yrf,Zrf,X,Y,Z,L)
    if debug:
        plotpot(tc.EL_RF,X,Y,Z,'1D plots','preflip EL_RF',[Irf,Jrf,Krf])
    tc.EL_RF=np.fliplr(tc.EL_RF)
    tc.EL_RF=np.flipud(tc.EL_RF)
    if debug: 
        plotpot(tc.EL_RF,X,Y,Z,'1D plots','EL_RF',[Irf,Jrf,Krf])
  
    #2) Expand the rf about its saddle point at the trapping position, save the quadrupole components.
    print 'Expanding RF about saddle point'
    [Xrf,Yrf,Zrf] = exactsaddle(tc.EL_RF,X,Y,Z,2,position) 
    [Irf,Jrf,Krf] = findsaddle(tc.EL_RF,X,Y,Z,2,position) 
    Qrf = spherharmxp(tc.EL_RF,Xrf+Xcorrection,Yrf+Xcorrection,Zrf,X,Y,Z,L)  
    tc.Qrf = 2*[Qrf[7][0]*3,Qrf[4][0]/2,Qrf[8][0]*6,-Qrf[6][0]*3,-Qrf[5][0]*3]
    tc.thetaRF = 45*((Qrf[8][0]/abs(Qrf[8][0])))-90*np.arctan((3*Qrf[7][0])/(3*Qrf[8][0]))/np.pi
    
    #3) Regenerate each DC electrode
    M1=np.zeros((N,NUM_DC)) 
    for el in range(0,NUM_DC): # do not include the constant term; el starts at 0 otherwise
        print ('Expanding DC Electrode {} ...'.format(el+1))
        if tc.electrodeMapping[el,1]:
            multipoleDCVoltages = np.zeros(NUM_DC)
            multipoleDCVoltages[el] = 1 
            Vdc = tf.potentials['EL_DC_{}'.format(el+1)]
#             if debug:
#                 plotpot(Vdc,X,Y,Z,'1D plots',('pre EL_{} DC Potential'.format(el+1)),'V (Volt)',[Irf,Jrf,Krf])
#             Vdc = dcpotential_instance(trap,multipoleDCVoltages,np.zeros(NUM_DC),np.zeros(NUM_DC),E) 
            if debug:
                plotpot(Vdc,X,Y,Z,'1D plots',('EL_{} DC Potential'.format(el+1)),'V (Volt)',[Irf,Jrf,Krf])
            print ('Applying correction to Electrode {} ...'.format(el+1))
            Q = spherharmxp(Vdc,Xrf+Xcorrection,Yrf+Ycorrection,Zrf,X,Y,Z,int(order[el]))                       
            print ('Regenerating Electrode {} potential...'.format(el+1))
            tf.potentials['EL_DC_{}'.format(el+1)]=spherharmcmp(Q,Xrf+Xcorrection,Yrf+Ycorrection,Zrf,X,Y,Z,int(order[el]))
            if debug:
                plotpot(Vdc,X,Y,Z,'1D plots',('post EL_{} DC Potential'.format(el+1)),'V (Volt)',[Irf,Jrf,Krf])
            print 'electrode',el+1
            check = np.real(Q[0:N].T)[0]
            iii = 0
            print check
            for e in check:
                a=abs(e)
                iii+=1
                if a>=0.1:
                    print iii

            M1[:,el] = Q[0:N].T
        elif tc.manualElectrodes[el]:
          multipoleDCVoltages = np.zeros(NUM_DC)
          manualDCVoltages = np.zeros(NUM_DC)
          manualDCVoltages[el]  = 1 
          print ('Building new trap instance for Electrolde {}...'.format(el+1))
          Vdc = dcpotential_instance(multipoleDCVoltages,manualDCVoltages,tc.manualElectrodes,E[0],E[1],E[2],NUM_DC,x,y,z)
          if debug:
              plotpot(Vdc,Irf,Jrf,Krf,'1D plots',sprintf('EL_{} DC Potential'.format(el)),'V (Volt)')
          print ('Applying correction to Electrode {} ...'.format(el+1))
          Q = spherharmxp(Vdc,Xrf+Xcorrection,Yrf+Ycorrection,Zrf,X,Y,Z,int(order[el]))                        
          print ('Regenerating Electrode {} potential...'.format(el+1))
          tf.potentials['EL_DC_{}'.format(el+1)] = spherharmcmp(Q,Xrf+Xcorrection,Yrf+Ycorrection,Zrf,X,Y,Z,int(order[el]))
    
    # Note: There used to be an auxilliary fuinction here that was not used: normalize.
    #4) Define the multipole Coefficients
    j = M1[0:N,:]
    tc.multipoleCoefficients = M1
    print('expand_field: Size of the multipole coefficient matrix is {}'.format(M1.shape))
    print('expand_field: ended successfully.')
    if save: 
        tc.expand_field=True
    tf.configuration=tc
    dataout=tf
    if save: 
        output = open(trap,'wb')
        pickle.dump(tf,output)
        output.close()
    return tf

def trap_knobs():
    """Updates trap.configuration with the matrix which controls the independent multipoles,
    and the kernel matrix. Start from the matrix multipoleCoefficients, return a field multipoleControl with
    the linear combimations of trap electrode voltages that give 1 V/mm, or 1 V/mm**2 of the multipole number i.
    Also return matrix multipoleKernel which is the kernel matrix of electrode linear combinations which do
    nothing to the multipoles.
    The order of multipole coefficients is:
    1/r0**[ x, y, z ] and 
    1/r0**2*[ (x^2-y^2)/2, (2z^2-x^2-y^2)/2, xy/2 yz/2 xz/2 ], where r0 is 1 mm
    (unless rescaling is applied)"""
    
    print('Executing trap_knobs...')
    
    #0) Define parameters
    from project_parameters import trap,position,debug,reg,name,save
    import numpy as np
    import matplotlib.pyplot as plt
    from all_functions import plotN
    import pickle, pprint
    file = open(trap,'rb')
    tf = pickle.load(file)
    #pprint.pprint(tf)
    file.close()
    V=tf.instance.DC 
    X=tf.instance.X
    Y=tf.instance.Y
    Z=tf.instance.Z
    
    #1) check to see what scripts have been run and build parameters from them
    if tf.configuration.expand_field!=True:
        return 'You must run expand_field first!'
    #if tf.configuration.trap_knobs==True:
        #return 'Already executed trap_knobs.'
    dataout = tf
    tc=tf.configuration
    multipoleCoefficients = tc.multipoleCoefficients # From expand_field (old regenthedata)
    print multipoleCoefficients.shape
    Mt=multipoleCoefficients[1:9,:] # cut out the first multipole coefficient (constant)
#     for i in range(Mt.shape[1]):
#         print Mt[:,i]
    numMultipoles = np.sum(tc.usedMultipoles) # number of multipoles; scalar
    C = np.zeros((numMultipoles,Mt.shape[1])) # intermediate array that may no longer be needed due to new spherharm ordering
    currentPos=0
    
    print numMultipoles
    #2) iterate through multipoles to build multipole controls
    for ii in range(numMultipoles):
        Mf = np.zeros((numMultipoles,1))
        Mf[ii] = 1
        P = np.linalg.lstsq(Mt,Mf)[0]
        print Mt
        print Mf
        print P
        Mout = np.dot(Mt,P) 
        err = Mf-Mout
        if debug:
            #fig=plt.figure()
            #plt.plot(err)
            #plt.title('Error of the fit elements')
            plotN(P[0:len(P)-1])
        C[ii,:] = P.T        

    #3) Helper function From: http://wiki.scipy.org/Cookbook/RankNullspace
    from numpy.linalg import svd
    def nullspace(A, atol=1e-13, rtol=0):
        """Compute an approximate basis for the nullspace of A.
        The algorithm used by this function is based on the singular value decomposition of `A`.
        
        Parameters
        ----------
        A : ndarray
            A should be at most 2-D.  A 1-D array with length k will be treated
            as a 2-D with shape (1, k)
        atol : float
            The absolute tolerance for a zero singular value.  Singular values
            smaller than `atol` are considered to be zero.
        rtol : float
            The relative tolerance.  Singular values less than rtol*smax are
            considered to be zero, where smax is the largest singular value.
    
        If both `atol` and `rtol` are positive, the combined tolerance is the
        maximum of the two; that is::
            tol = max(atol, rtol * smax)
        Singular values smaller than `tol` are considered to be zero.
    
        Return value
        ------------
        ns : ndarray
            If `A` is an array with shape (m, k), then `ns` will be an array
            with shape (k, n), where n is the estimated dimension of the
            nullspace of `A`.  The columns of `ns` are a basis for the
            nullspace; each element in numpy.dot(A, ns) will be approximately
            zero.
        """
        A = np.atleast_2d(A)
        u, s, vh = svd(A)
        tol = max(atol, rtol * s[0])
        nnz = (s >= tol).sum()
        ns = vh[nnz:].conj().T
        return ns
        
    K = nullspace(Mt)
    print K.shape

    #4) regularize C with K
    if reg:
        for ii in range(numMultipoles):
            Cv = C[ii,:].T
            Lambda = np.linalg.lstsq(K,Cv)[0]
            test=np.dot(K,Lambda)
            C[ii,:] = C[ii,:]-test
    
    #5) update instance configuration
    tc.multipoleKernel = K
    print C.T.shape
    tc.multipoleControl = C.T
    tc.trap_knobs = True
    dataout.configuration=tc
    
    if save: 
        import pickle
        #name=name+'.pkl'
        print ('Saving '+name+' as a data structure...')
        output = open(trap,'wb')
        pickle.dump(dataout,output)
        output.close()
    
    return 'Completed trap_knobs.'

def set_dc():
    """Provides the DC voltages for all DC electrodes to be set to. 
    Uses parameters and voltage controls from analyze_trap.
    Output an array of values to set each electrode; used as VMULT for dcpotential_instance in ppt3.
    The Ui and Ei values control the weighting of each term of the multipole expansion.
    Nikos, July 2009
    Cleaned up October 2013
    William Python 2014""" 
    
    #0) set parameters
    from analyze_trap import trap,multipoleControls,regularize,frequencyRF,E,U1,U2,U3,U4,U5,ax,az,phi
    import numpy as np
    import pickle, pprint
    file = open(trap,'rb')
    tf = pickle.load(file)
    #pprint.pprint(tf)
    file.close()
    V=tf.instance.DC 
    X=tf.instance.X
    Y=tf.instance.Y
    Z=tf.instance.Z
    tc=tf.configuration
    el = []
    
    #1) check if trap_knobs has been run yet, creating multipoleControl and multipoleKernel
    if tc.trap_knobs != True:
        return 'WARNING: You must run trap_knobs first!'

    #2a) determine electrode voltages directly
    elif multipoleControls: # note plurality to contrast from attribute
        inp = np.array([E[0], E[1], E[2], U1, U2, U3, U4, U5]).T
        el = np.dot(tc.multipoleControl,inp)     # these are the electrode voltages
     
    #2b) determine electrode volages indirectly
    else:
        charge = tc.charge
        mass = tc.mass
        V0 = mass*(2*np.pi*frequencyRF)**2/charge
        U2 = az*V0/8
        U1 = U2+ax*V0/4
        U3 = 2*U1*np.tan(2*np.pi*(phi+tc.thetaRF)/180)
        U1p= np.sqrt(U1**2+U3**2/2)
        U4 = U1p*tc.Qrf[4]/tc.Qrf[1]
        U5 = U1p*tc.Qrf[5]/tc.Qrf[1]
        inp = np.array([E[0], E[1], E[2], U1, U2, U3, U4, U5]).T
        mCf = tc.multipoleCoefficients[1:9,:]
        el = np.dot(mCf.T,inp) # these are the electrode voltages
        el = np.real(el)
        
    #3) regularize if set to do so
    if regularize: 
        C = el
        Lambda = np.linalg.lstsq(tc.multipoleKernel,C)
        Lambda=Lambda[0]
        el = el-(np.dot(tc.multipoleKernel,Lambda))
        
    return el

def ppt3():
    """A post processing tool that analyzes the trap. This is the highest order function.
    It plots an input trap in a manner of ways and returns the frequencies and positions determined by pfit.
    Before 2/19/14, it was far more complicated. See ppt2 for the past version and variable names.
    All necessary configuration parameters should be defined by dcpotential instance, trap knobs, and so on prior to use.
    
    Change rfplot and dcplot to control the "dim" input to plotpot for plotting the potential fields.
    
    There is also an option to run findEfield. This determines the stray electric field for given DC voltages.
    
    Nikos, January 2009
    William Python Feb 2014
    """ 
    #################### 0) assign internal values #################### 
    from project_parameters import manualElectrodes,weightElectrodes,save,debug
    from project_parameters import trap,driveAmplitude,driveFrequency,findEfield,justAnalyzeTrap,rfplot,dcplot
    from analyze_trap import E,U1,U2,U3,U4,U5,ax,az,phi
    from all_functions import findsaddle,exactsaddle,plotpot,dcpotential_instance,d_e,pfit,spherharmxp
    import numpy as np
    import pickle, pprint
    file = open(trap,'rb')
    tf = pickle.load(file)
    #pprint.pprint(tf)
    file.close()
    VMULT = set_dc()
    VMAN = manualElectrodes
    IMAN = weightElectrodes
    tf.instance.DC = dcpotential_instance(trap,VMULT,VMAN,IMAN,E,True) 
    V=tf.instance.DC                # old VELDC
    X=tf.instance.X                 # grid vector
    Y=tf.instance.Y                 # grid vector
    Z=tf.instance.Z                 # grid vector
    RFampl = driveAmplitude         # drive amplitude of RF
    f = driveFrequency              # drive frequency of RF
    Omega = 2*np.pi*f               # angular frequency of RF
    # trap configuration attributes
    tc = tf.configuration                       # trap configuration shorthand
    qe = tc.charge                              # elementary charge unit in SI
    mass = tc.mass                              # mass of the ion
    Zval = tc.position                          # position of ion on z-axis
    V0 = mass*(2*np.pi*f)**2/qe
    out = tc                                    # for quality checking at end of function; may no longer be needed
    data = tf.potentials                        # shorthand for refernce to trapping field potentials; mostly RF
    [x,y,z] = np.meshgrid(X,Y,Z)

    #################### 1) analyze the RF potential #################### 
    [Irf,Jrf,Krf] = findsaddle(data.EL_RF,X,Y,Z,2,Zval)
    Vrf = RFampl*data.EL_RF
    
    plotpot(V,X,Y,Z,rfplot,'initial V','V_{rf} (Volt)',[Irf,Jrf,Krf])
    
    plotpot(Vrf,X,Y,Z,rfplot,'RF potential','V_{rf} (Volt)',[Irf,Jrf,Krf])
    if E == None:      # check the initial guess of the E field
        return 'There is no E field. Create an instance with dcpotential_instance.'
    else:
        Vdc = dcpotential_instance(trap,VMULT,VMAN,IMAN,E)                                                         
        [Idum,Jdum,Kdum] =  findsaddle(Vdc,X,Y,Z,2,Zval)
        plotpot(Vdc,X,Y,Z,dcplot,'DC potential (stray field included)','V_{dc} (Volt)',[Idum,Jdum,Kdum])
        
    #################### 2) findEfield ####################
    def findE(Efield,x,y,z,X,Y,Z):
        """Temporarily a helper function. Will likely become this entirely if not simply cut from ppt3."""
        from all_functions import d_e,findsaddle,plotpot,dcpotential_instance
        Vdc = dcpotential_instance(trap,VMULT,VMAN,IMAN,E)  
        Efield=[1,1,1]
        if np.sum(Efield)==0:
            E0=[0,0,0]
            if debug!=True:
                    stx = raw_input('Give an initial guess for stray Ex field (in V/m).')
                    sty = raw_input('Give an initial guess for stray Ey field (in V/m).')
                    stz = raw_input('Give an initial guess for stray Ey field (in V/m).')
                    E0x = float(stx)/1e3 # Convert to mV. #Matlab: E0 = sscanf(st,'%f',inf)'/1e3
                    E0y = float(sty)/1e3
                    E0z = float(stz)/1e3
                    E0=[E0x,E0y,E0z]
            else:
                E0=[0,0,0]
            dist0 = d_e(E0,Vdc,data,x,y,z,X,Y,Z,Zval)
            Vdum = dcpotential_instance(trap,VMULT,VMAN,IMAN,E0[0],E0[1],E0[2])   
            [Idum,Jdum,Kdum] =  findsaddle(Vdum,X,Y,Z,2,Zval)
            plotpot(Vdum,X,Y,Z,'1D plots','Initial guess for DC potential','V_{dc} (Volt)',[Irf,Jrf,Krf])
            st = raw_input('Happy (y/n)?\n')
            if st=='y': 
                print 'findEfield complete'
                return dist0
            else: 
                return findE(Efield,x,y,z,X,Y,Z)
        else:
            E0 = Efield
            dist0 = d_e(E0,Vdc,data,x,y,z,X,Y,Z,Zval)
            Vdum = dcpotential_instance(trap,VMULT,VMAN,IMAN,E0[0],E)
            [Idum,Jdum,Kdum] =  findsaddle(Vdum,X,Y,Z,2,Zval)
            plotpot(Vdum,X,Y,Z,'1D plots','Initial guess for DC potential','V_{dc} (Volt)',[Idum,Jdum,Kdum])
            print 'findEfield complete'
            return dist0
    # actually find E field
    if findEfield: # this option means find stray field
        E0=Efield
        #this is the main code that calls d_e; not too important
        dist0=findE(Efield,x,y,z,X,Y,Z)
        print ('Initial guess for stray field: ({0},{1},{2}) V/m.'.format(1e3*E0[0],1e3*E0[1],1e3*E0[2]))
        print ('Miscompensation in the presence of this field: {} micron.'.format(1e3*dist0))
        print ('Optimizing stray field value...\n')
        import scipy.optimize as spo
        E=spo.minimize(d_e,E0,args=(Vdc,data,x,y,z,X,Y,Z,Zval)) 
        E=E.x #Unpack desired value.
        dist = d_e(E,Vdc,data,x,y,z,X,Y,Z,Zval)
        print('Stray field is ({0},{1},{2}) V/m.'.format(1e3*E[0],1e3*E[1],1e3*E[2]))
        print('With this field the compensation is optimized to {} micron.'.format(1e3*dist))
        if dist>5e-3:
            params.E = []
            print ('Miscompensation larger than 5 micron. Repeating.\n')
        return 
    #################### 3) justAnalyzeTrap ####################
    else: # this option means do not optimize anything, and just analyze the trap
        print('Running ppt3 in plain analysis mode (no optimizations).')
        dist = np.NaN
        dist = d_e(E,Vdc,data,x,y,z,X,Y,Z,Zval)
        print ('Stray field is ( {0}, {1}, {2}) V/m.'.format(1e3*E[0],1e3*E[1],1e3*E[2]))
        print ('With this field, the compensation is optimized to {} micron.'.format(1e3*dist))
    Vdc = dcpotential_instance(trap,VMULT,VMAN,IMAN,E)
    # RF should be dim 2, DC dim 3
    [XRF,YRF,ZRF] = exactsaddle(data.EL_RF,X,Y,Z,2,Zval)  
    [XDC,YDC,ZDC] = exactsaddle(Vdc,X,Y,Z,3,Zval)
    # for debugging purposes
    #[XRF,YRF,ZRF] = findsaddle(data.EL_RF,X,Y,Z,2,Zval)
    #[XDC,YDC,ZDC] = findsaddle(Vdc,X,Y,Z,3,Zval)
    print ('RF saddle: ({0},{1},{2})\nDC saddle ({3},{4},{5}).'.format(XRF,YRF,ZRF,XDC,YDC,ZDC))
    plotpot(Vdc,X,Y,Z,dcplot,'Compensated DC potential','V_{dc} (V)',[Irf,Jrf,Krf])
    [IDC,JDC,KDC] = findsaddle(Vdc,X,Y,Z,2,Zval)
    [fx,fy,fz,theta,Depth,rx,ry,rz,xe,ye,ze,superU] = pfit(trap,E,f,RFampl)
    Qrf = spherharmxp(Vrf,XRF,YRF,ZRF,X,Y,Z,2)           
    if np.sqrt((XRF-XDC)**2+(YRF-YDC)**2+(ZRF-ZDC)**2)>0.008: 
        print 'Expanding DC with RF'
        Qdc = spherharmxp(Vdc,XRF,YRF,ZRF,X,Y,Z,2) 
    else:
        print 'Expanding DC'
        Qdc = spherharmxp(Vdc,XDC,YDC,ZDC,X,Y,Z,2) 
    # Sanity testing; quality check no longer used
    Arf = 2*np.sqrt( (3*Qrf[7])**2+(3*Qrf[8])**2 )
    Thetarf = 45*(Qrf[8]/abs(Qrf[8]))-90*np.arctan((3*Qrf[7])/(3*Qrf[8]))/np.pi
    Adc = 2*np.sqrt( (3*Qdc[7])**2+(3*Qdc[8])**2 )
    Thetadc = 45*(Qrf[8]/abs(Qrf[8]))-90*np.arctan((3*Qdc[7])/(3*Qdc[8]))/np.pi
    out.E = E
    out.miscompensation = dist
    out.ionpos = [XRF,YRF,ZDC]
    out.ionposIndex = [Irf,Jrf,Krf]
    out.f = [fx,fy,fz]
    out.theta = theta
    out.trapdepth = Depth/qe 
    out.escapepos = [xe,ye,ze]
    out.Quadrf = 2*np.array([Qrf[7]*3,Qrf[4]/2,Qrf[8]*6,-Qrf[6]*3,-Qrf[5]*3])
    out.Quaddc = 2*np.array([Qdc[7]*3,Qdc[4]/2,Qdc[8]*6,-Qdc[6]*3,-Qdc[5]*3])
    out.Arf = Arf
    out.Thetarf = Thetarf
    out.Adc = Adc
    out.Thetadc = Thetadc
    T = np.array([[2,-2,0,0,0],[-2,-2,0,0,0],[0, 4,0,0,0],[0, 0,1,0,0],[0, 0,0,1,0],[0, 0,0,0,1]])
    Qdrf = out.Quadrf.T
    Qddc = out.Quaddc.T
    out.q = (1/V0)*T*Qdrf
    out.alpha = (2/V0)*T*Qddc
    out.Error = [X[IDC]-XDC,Y[JDC]-YDC,Z[KDC]-ZDC]
    out.superU = superU
    
    # update the trapping field data structure with instance attributes
    tf.configuration=out
    tf.instance.driveAmplitude = driveAmplitude
    tf.instance.driveFrequency = driveFrequency
    tf.instance.E = E
    tf.instance.U1 = U1
    tf.instance.U2 = U2
    tf.instance.U3 = U3
    tf.instance.U4 = U4
    tf.instance.U5 = U5
    tf.instance.ax = ax
    tf.instance.az = az
    tf.instance.phi = phi
    tf.instance.ppt = True

    if save==True:
        import pickle
        update=trap
        print update
        print ('Saving '+update+' as a data structure...')
        output = open(update,'wb')
        pickle.dump(tf,output)
        output.close()

    return #out # no output needed really

print('Referencing all_functions...')
# Secondary Functions
def dcpotential_instance(trap,VMULT,VMAN,IMAN,E,update=None):
    """ Calculates the dc potential given the applied voltages and the stray field.
    Creates a third attribute of trap, called instance, a 3D matrix of potential values
    
    trap: file path and name, including '.pkl'
    VMULT: electrode voltages determined by the multipole control algorithm
    VMAN: electrode voltages determined by manual user control 
          e.g. VMAN  = [0,0,-2.,0,...] applies -2 V to electrode 3
    IMAN: array marking by an entry of 1 the electrodes which are under manual control, 
          e.g. IMAN = [0,0,1,0,...] means that electrode 3 is under manual control
    BOTH above conditions are necessary to manually apply the -2 V to electrode 3
    
    Ex,Ey,Ez: stray electric field; 3D matrices
    
    update: name to save new file as; typically the same as the name used for get_trap
    
    Nikos, cleaned up June 2013
    William, converted to Python Jan 2014"""
    
    print trap,VMULT,VMAN,IMAN,E
    
    import pickle, pprint
    file = open(trap,'rb')
    tf = pickle.load(file)
    #pprint.pprint(tf)
    file.close()
    p=tf.potentials # shorthand to refer to all potentials
    nue=tf.configuration.numUsedElectrodes
    X,Y,Z=tf.potentials.X,tf.potentials.Y,tf.potentials.Z # grid vectors
    import numpy as np
    x,y,z=np.meshgrid(X,Y,Z)   
    [Ex,Ey,Ez]=E
    Vout = np.zeros((p['EL_DC_1'].shape[0],p['EL_DC_1'].shape[1],p['EL_DC_1'].shape[2]))
    
    # build up the potential from the manual DC electrodes
    for ii in range(nue-1):
        if int(IMAN[ii])==1:
            Vout = Vout + VMAN[ii]*p['mEL_DC_{}'.format(ii+1)]
    # build up the potential from the normal DC elctrodes
    for ii in range(nue-1):
        Vout = Vout + VMULT[ii]*p['EL_DC_{}'.format(ii+1)]
    # subtract the stray E field
    Vout = Vout-Ex*x-Ey*y-Ez*z
    
    # update the trapping field data structure with instance attributes
    tf.instance.DC=Vout
    tf.instance.RF=p.EL_RF # not needed, but may be useful notation
    tf.instance.X=X
    tf.instance.Y=Y
    tf.instance.Z=Z
    tf.instance.check=True
    if update==True:
        name=trap
        print ('Saving '+name+' as a data structure...')
        output = open(name,'wb')
        pickle.dump(tf,output)
        output.close()

    return tf.instance.DC

def d_e(Ei,Vdc,data,x,y,z,X,Y,Z,Zval):
    """find the miscompensation distance, d_e, for the rf and dc potential 
    given in the main program, in the presence of stray field Ei"""
    from all_functions import exactsaddle
    import numpy as np 
    dm = Ei
    E1 = dm[0]
    E2 = dm[1]
    E3 = dm[2]
    Vl = Vdc-E1*x-E2*y-E3*z
    from all_functions import plotpot
    print plotpot(data.EL_RF,X,Y,Z,'1D plots','title','ylab',[0,0,0])
    [Xrf,Yrf,Zrf] = exactsaddle(data.EL_RF,X,Y,Z,2,Zval)
    #[Xdc,Ydc,Zdc] = exactsaddle(data.EL_RF,X,Y,Z,3) 
    [Idc,Jdc,Kdc] = findsaddle(data.EL_RF,X,Y,Z,3) # no saddle point with exact
    Xdc,Ydc,Zdc=X[Idc],Y[Jdc],Z[Kdc]
    f = np.sqrt((Xrf-Xdc)**2+(Yrf-Ydc)**2+(Zrf-Zdc)**2)
    return f

def pfit(trap,E,driveFrequence,driveAmplitude):
    """find the secular frequencies, tilt angle, and position of the dc 
    saddle point for given combined input parameters. 
    
    fx,fy,fz are the secular frequencies
    theata is the angle of rotation from the p2d transformation (rotation)
    Depth is the distance between the potential at the trapping position and at the escape point
    Xdc,Ydc,Zdc are the coordinates of the trapping position
    Xe,Ye,Ze are the coordinates of the escape position
    
    
    The stray field E was originally defined in ppt2 when this was a helper function to it.
    Now it is called here. Are they not the same?
    William Python February 2014."""
    
    #0) open trap
    from analyze_trap import trap
    import numpy as np
    import pickle, pprint
    file = open(trap,'rb')
    tf = pickle.load(file)
    #pprint.pprint(tf)
    file.close()

    #1) find dc potential
    from all_functions import set_dc,plotpot,exactsaddle,findsaddle,p2d,trapdepth
    from project_parameters import dcVoltages,manualElectrodes,weightElectrodes,mp,qe,debug,driveFrequency,driveAmplitude
    dcVoltages=set_dc() #should this be set_dc or from analyze_trap? U is 0 with but no saddle without.
    VL = dcpotential_instance(trap,dcVoltages,manualElectrodes,weightElectrodes,E)
    X=tf.instance.X
    Y=tf.instance.Y
    Z=tf.instance.Z
    Zval=tf.configuration.position
    #[Xdc,Ydc,Zdc] = exactsaddle(VL,X,Y,Z,3) # debug comment out
    [Idc,Jdc,Kdc] = findsaddle(VL,X,Y,Z,3)
    [Xdc,Ydc,Zdc]=[X[Idc],Y[Jdc],Z[Kdc]]
    [Irf,Jrf,Krf] = findsaddle(tf.potentials.EL_RF,X,Y,Z,2,Zval)
    mass=mp
    Omega=2*np.pi*driveFrequency
    e=qe
    
    #2) find pseudopotential
    """Gebhard, Oct 2010:
    changed back to calculating field numerically in ppt2 instead directly
    with bemsolver. this is because the slow bemsolver (new version) does not output EX, EY, EZ."""
    Vrf = driveAmplitude*tf.potentials.EL_RF 
    [Ex,Ey,Ez] = np.gradient(Vrf)
    Esq1 = Ex**2 + Ey**2 + Ez**2
    Esq = (driveAmplitude*1e3*tf.potentials.EL_RF)**2 

    #3) plotting pseudopotential, etc; outdated?
    PseudoPhi = Esq1/(4*mass*Omega**2) 
    print 'Pseudo: ',np.amax(PseudoPhi)
    plotpot(PseudoPhi,X,Y,Z,'1D plots','Pseudopotential','U_{ps} (eV)',[Irf,Jrf,Krf])
    
    print 'VL: ',np.amax(VL)
    plotpot(VL,X,Y,Z,'1D plots','VL','U_{sec} (eV)',[Irf,Jrf,Krf])
    U = PseudoPhi*10**-6+VL*10**4 # total trap potential
    superU = U
    print 'TrapPotential: ',np.amax(U)
    plotpot(U,X,Y,Z,'1D plots','TrapPotential','U_{sec} (eV)',[Irf,Jrf,Krf])
    #[I,J,K] = findsaddle(U/np.amax(U),X,Y,Z,2,Zval) # ???
    plotpot(tf.potentials.EL_RF,X,Y,Z,'1D plots','RF potential','(eV)',[Irf,Jrf,Krf])
    
    #4) determine trap frequencies and tilt in radial directions
    Uxy = U[Irf-3:Irf+3,Jrf-3:Jrf+3,Krf]
    MU = np.amax(Uxy)
    x,y,z=np.meshgrid(X,Y,Z)
    dL = (y[Irf+3,Jrf,Krf]-y[Irf,Jrf,Krf]) # is this X? Originally x. Temporarily y so that dL not 0.
    Uxy = Uxy/MU
    xr = (x[Irf-3:Irf+3,Jrf-3:Jrf+3,Krf]-x[Irf,Jrf,Krf])/dL 
    yr = (y[Irf-3:Irf+3,Jrf-3:Jrf+3,Krf]-y[Irf,Jrf,Krf])/dL
    [C1,C2,theta] = p2d(Uxy,xr,yr)                       
    fx = (1e3/dL)*np.sqrt(2*C1*MU/(mass))/(2*np.pi)
    fy = (1e3/dL)*np.sqrt(2*C2*MU/(mass))/(2*np.pi)
    
    #5) trap frequency in axial direction
    MU = 1
    Uz=U[Irf,Jrf,:]/MU 
    l1 = np.max([Krf-6,1])
    l2 = np.min([Krf+6,Z.shape[0]])
    p = np.polyfit((Z[l1:l2]-Z[Krf])/dL,Uz[l1:l2],6)
    ft = np.polyval(p,(Z-Z[Krf])/dL)
    Zt=((Z[l1:l2]-Z[Krf])/dL).T
    Uzt=Uz[l1:l2].T
    if debug:
        import matplotlib.pyplot as plt
        fig=plt.figure()
        plt.plot(Z,MU*Uz)
        plt.plot(Z[l1:l2],MU*ft[l1:l2],'r')
        plt.title('Potential in axial direction')
        plt.xlabel('axial direction (mm)')
        plt.ylabel('trap potential (J)')
        plt.show()
    fz = [(1e3/dL)*np.sqrt(2*p[5]*MU/(mass))/(2*np.pi)]
    [Depth,Xe,Ye,Ze] = trapdepth(U,X,Y,Z,Irf,Jrf,Krf,debug=True) # divide by e to make U larger?     
         
    return [fx,fy,fz,theta,Depth,Xdc,Ydc,Zdc,Xe,Ye,Ze,superU] 

def exactsaddle(V,X,Y,Z,dim,Z0=None):
    """This version finds the approximate saddle point using pseudopotential,
    does a multipole expansion around it, and finds the exact saddle point by
    maximizing the quadrupole terms. Similar to interpolation.
    
    V is a 3D matrix containing an electric potential and must solve Laplace's equation
    X,Y,Z are the vectors that define the grid in three directions

    dim is the dimensionality (2 or 3).
    Z0 is the coordinate where a saddle point will be sought if dim==2.
    Nikos Daniilidis 9/1/09.
    Had issues with function nesting and variable scope definitions in Octave.
    Revisited for Octave compatibility 5/25/13.
    Needs Octave >3.6, pakages general, miscellaneous, struct, optim. 
    William Python Jan 2014"""
    
    import numpy as np
    import scipy.optimize as spo
    from all_functions import findsaddle,sumofefield

    if dim==3:
        [I,J,K]=findsaddle(V,X,Y,Z,3) # guess saddle point; Z0 not needed
        print I,J,K
        if I<2 or I>V.shape[0]-2: 
            print('exactsaddle.py: Saddle point out of bounds in radial direction.')
            return
        if J<2 or J>V.shape[1]-2:
            print('exactsaddle.py: Saddle point out of bounds in vertical direction.')
            return
        if K<2 or K>V.shape[2]-2:
            print('exactsaddle.py: Saddle point out of bounds in axial direction.')
            return
        Vn = V[I-2:I+3,J-2:J+3,K-2:K+3] # create smaller 5x5x5 grid around the saddle point to speed up optimization
        # note that this does not prevent the optimization function from trying values outside this
        Xn,Yn,Zn=X[I-2:I+3],Y[J-2:J+3],Z[K-2:K+3] # change grid vectors as well
        #################################### Minimize
        r0=[X[I],Y[J],Z[K]]
        r=spo.minimize(sumofefield,r0,args=(Vn,Xn,Yn,Zn)) 
        r=r.x # unpack for desired values
        Xs,Ys,Zs=r[0],r[1],r[2] 
    #################################################################################################    
    if dim==2: 
        if len(V.shape)==3:
            K=0 # in case there is no saddle
            for i in range(len(Z)):
                if Z[i-1]<Z0 and Z[i]>=Z0:
                    K=i-1
        Vs = V.shape
        if K>=Vs[1]: # Matlab had Z, not V; also changed from == to >=
            return('The selected coordinate is at the end of range.')
        v1=V[:,:,K-1] # potential to left
        v2=V[:,:,K] # potential to right (actually right at estimate; K+1 to be actually to right)
        V2=v1+(v2-v1)*(Z0-Z[K-1])/(Z[K]-Z[K-1]) # averaged potential around given coordinate
        [I,J,K0]=findsaddle(V,X,Y,Z,2,Z0) # should be K instead of Z0? 
        print I,J,K0
        if (I<2 or I>V.shape[0]-2): 
            print('exactsaddle.py: Saddle point out of bounds in radial direction.\n')
            return
        if (J<2 or J>V.shape[1]-2):
            print('exactsaddle.py: Saddle point out of bounds in vertical direction.\n')
            return
        # This is not needed? Causes problems if I,J are too large (saddle point near edge).
#         A = np.zeros((5,5,5))
#         for i in range(5): # What is this used for? # Matlab 9, not 4
#             A[:,:,i]=V2[I-2:I+3,J-2:J+3] # Matlab 4, not 2
        Vn = V[I-2:I+3,J-2:J+3,K-2:K+3] # create smaller 5x5x5 grid around the saddle point to speed up optimization
        # note that this does not prevent the optimization function from trying values outside this
        Xn,Yn,Zn=X[I-2:I+3],Y[J-2:J+3],Z[K-2:K+3] # Matlab 4, not 2
        ################################## Minimize
        r0=X[I],Y[J],Z0
        print r0
        r=spo.minimize(sumofefield,r0,args=(Vn,Xn,Yn,Zn)) 
        r=r.x # unpack for desired values
        Xs,Ys,Zs=r[0],r[1],Z0
    return [Xs,Ys,Zs]

def findsaddle(V,X,Y,Z,dim,Z0=None):
    """Returns the indices of the local extremum or saddle point of the scalar A as (Is,Js,Ks).
    V is a 3D matrix containing an electric potential and must solve Laplace's equation
    X,Y,Z are the vectors that define the grid in three directions
    
    Z0 is the Z axis index (may be a decimal) on which the saddle point is evaluated, if dim==2. 
    
    3/15/14: Z0 is coord, not index; Ks is the index
    
    For dim==2, the values of A are linearly extrapolated from [Z0] and [Z0]+1
    to those corresponding to Z0 and Ks is such that z[Ks]<Z0, z[Ks+1]>=Z0."""
    debug=False # internal code only; typically False
    import numpy as np
    import matplotlib.pyplot as plt
    if (dim==2 and Z0==None):
        return 'z0 needed for evaluation'
    
    if dim==3:
        if len(V.shape)!=3:
            return('Problem with findsaddle.m dimensionalities.')
        f=V/float(np.amax(V)) # Normalize field
        [Ex,Ey,Ez]=np.gradient(f) # grid spacing is automatically consistent thanks to BEM-solver
        E=np.sqrt(Ex**2+Ey**2+Ez**2) # magnitude of gradient (E field)
        print E
        m=E[1,1,1]
        origin=[1,1,1]
        for i in range(E.shape[0]):
            for j in range(E.shape[1]):
                for k in range(E.shape[2]):
                    if E[i,j,k]<m:
                        m=E[i,j,k]
                        origin=[i,j,k]          
        if debug:
            print('DEBUGGING...')
            fig=plt.figure()
            e=np.reshape(E,(1,E.shape[0]*E.shape[1]*E.shape[2]))
            ind,e=np.argsort(e),np.sort(e)
            e=e[0]
            ind=ind[0] #Sort V by the same indexing.
            v=np.reshape(V,(1,V.shape[0]*V.shape[1]*V.shape[2]))
            v=v[0]
            plt.plot(e/float(np.amax(e)))
            # helper function
            def index_sort(v,e):
                """Takes in two lists of the same length and returns the first sorted by the indexing of i sorted."""
                es=np.sort(e)
                ix=np.argsort(e)
                vs=np.ones(len(v)) #Sorted by the sorting defined by f being sorted. 
                # If v==e, this returns es.
                for i in range(len(v)):
                    j=ix[i]
                    vs[i]=v[j]
                return vs
            
            v=index_sort(v,e) # Is it supposed to look like this?
            plt.plot(v/float(np.amax(v)))
            plt.title('Debugging: blue is sorted gradient, green is potential sorted by gradient')
            plt.show() #f is blue and smooth, v is green and fuzzy.
        if origin[0]==(1 or V.shape[0]):
            print('findsaddle: Saddle out of bounds in  x (i) direction.\n')
            return
        if origin[0]==(1 or V.shape[1]):
            print('findsaddle: Saddle out of bounds in  y (j) direction.\n')
            return
        if origin[0]==(1 or V.shape[2]): 
            print('findsaddle: Saddle out of bounds in  z (k) direction.\n')
            return
    #################################################################################################
    if dim==2: # Extrapolate to the values of A at z0.
        V2=V
        if len(V.shape)==3:
            Ks=0 # in case there is no saddle point
            for i in range(len(Z)):
                if Z[i-1]<Z0 and Z[i]>=Z0:
                    Ks=i-1
                    if Z0<1:
                        Ks+=1
            Vs=V.shape
            if Ks>=Vs[1]: # Matlab had Z, not V; also changed from == to >=
                return('The selected coordinate is at the end of range.')
            v1=V[:,:,Ks] 
            v2=V[:,:,Ks+1]
            V2=v1+(v2-v1)*(Z0-Z[Ks])/(Z[Ks+1]-Z[Ks])
        V2s=V2.shape
        if len(V2s)!=2: # Old: What is this supposed to check? Matlab code: (size(size(A2),2) ~= 2)
            return('Problem with findsaddle.py dimensionalities. It is {}.'.format(V2s))
        f=V2/float(np.max(abs(V2)))
        [Ex,Ey]=np.gradient(f)
        E=np.sqrt(Ex**2+Ey**2)
        m=float(np.min(E))
        if m>1e-4: # This requires a grid with step size 0.01, not just 0.1.
            if debug:
                Is,Js=np.NaN,np.NaN
                print('Warning, there seems to be no saddle point.')
        mr=E[0,0]
        Is,Js=1,1 # in case there is no saddle
        for i in range(E.shape[0]):
            for j in range(E.shape[1]):
                if E[i,j]<mr:
                    mr=E[i,j]
                    Is=i
                    Js=j
        if Is==1 or Is==V.shape[0]:
            print('findsaddle: Saddle out of bounds in  x (i) direction.\n')
            return
        if Js==1 or Js==V.shape[1]:
            print('findsaddle: Saddle out of bounds in  y (j) direction.\n')
            return
        origin=[Is,Js,Ks]
    return origin

def meshslice(V,n,X,Y,Z): 
    """Plots successive slices of matrix V in the direction given by n.
    n=1[I],2[J],3[K]
    x,y,z are vectors that define the grid in three dimensions
    William Python Jan 2014
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import mpl_toolkits.mplot3d.axes3d as p3
    import time
    
    order=np.array([(1,2,0),(2,0,1),(0,1,2)])
    q=np.transpose(V,(order[n])) # See projection for why we could also use take instead.
    if n==0: # Make a less cumbersome and more consistent version of this?
        i,j=X,Y
        i,j=np.array([i]),np.array([j]).T
        I,J=i,j
        for m in range(j.shape[0]-1): # -1 because we already have the first row as I.
            I=np.vstack((I,i))
        for m in range(i.shape[1]-1):
            J=np.hstack((J,j))
    if n==1:
        i,j=Y,Z
        i,j=np.array([i]),np.array([j]).T
        I,J=i,j
        for m in range(j.shape[0]-1): # -1 because we already have the first row as I.
            I=np.vstack((I,i))
        for m in range(i.shape[1]-1):
            J=np.hstack((J,j))
    if n==2:
        i,j=Z,X
        i,j=np.array([i]),np.array([j]).T
        I,J=i,j
        for m in range(j.shape[0]-1): # -1 because we already have the first row as I.
            I=np.vstack((I,i))
        for m in range(i.shape[1]-1):
            J=np.hstack((J,j))
    labels={
        0:('horizontal axial (mm)','height (mm)'),
        1:('horizontal radial (mm)','horizontal axial (mm)'),
        2:('height (mm)','horizontal radial (mm)')
        }   
    class animated(object): # 4D, plots f(x,y,z0) specific to meshslice.
        def __init__(self,I,J,q):
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.I,self.J=I,J
            self.q=q[:,0,:]
            self.surf=self.ax.plot_surface(self.J,self.I,self.q,cmap=cm.coolwarm,antialiased=False)
        def drawNow(self,ii,q,n):
            self.surf.remove()
            self.slc=q[:,ii,:]
            self.surf=self.ax.plot_surface(self.J,self.I,self.slc,cmap=cm.coolwarm,antialiased=False)
            plt.ylabel(labels[n][1])
            plt.xlabel(labels[n][0])
            #plt.title(ii) #Optional: this moves down during animation.
            plt.draw() # redraw the canvas
            time.sleep(0.1)
            self.fig.show()
    anim=animated(I,J,q)
    for ii in range(q.shape[1]):
        if ii==q.shape[1]-1:
            plt.title('Animation complete.')
        anim.drawNow(ii,q,n)
    return plt.show()

def plotpot(V,X,Y,Z,key='1D plots',tit=None,ylab=None,origin=None): 
    """V is a 3D matrix containing an electric potential and must solve Laplace's equation
    X,Y,Z are the vectors that define the grid in three directions
    
    Makes 2D mesh plots and 1D plots of 3D matrix around an origin
    key: 0: no plots, 1: 2D plots, 2: 1D plots, 3: both
    tit is the title of the plots to be produced.
    ylab is the label on the y axis of the plot produced.
    William Python Jan 2014"""
    
    import matplotlib.pyplot as plt
    print 'running plotpot...',key
    if origin==None:
        from all_functions import exactsaddle,findsaddle
        #origin=exactsaddle(V,X,Y,Z,3) # these are coordinates, not indices; findsaddle instead
        origin=findsaddle(V,X,Y,Z,3)
    if (key==0 or key=='no plots'):
        return 
    if (key==1 or key=='2D plots' or key==3 or key=='both'): # 2D Plots, animation
        from all_functions import meshslice
        meshslice(V,0,X,Y,Z) 
        meshslice(V,1,X,Y,Z) 
        meshslice(V,2,X,Y,Z) 
    if (key==2 or key=='1D plots' or key==3 or key=='both'): # 1D Plots, 3 subplots
        ########## Plot I ##########
        axis=X
        projection=V[:,origin[1],origin[2]]
        fig=plt.figure()
        plt.subplot(2,2,1)
        plt.plot(axis,projection) 
        plt.title(tit)
        plt.xlabel('x (mm)')
        plt.ylabel('Interpreter')
        ######### Plot J ##########
        axis=Y
        projection=V[origin[0],:,origin[2]]
        #fig=plt.figure() # for individual plots instead of subplots
        plt.subplot(2,2,2)
        plt.plot(axis,projection)
        #plt.title(tit)
        plt.xlabel('y (mm)')
        plt.ylabel('Interpreter')
        ######### Plot K ##########
        axis=Z
        projection=V[origin[0],origin[1],:]
        #fig=plt.figure()
        plt.subplot(2,2,3)
        plt.plot(axis,projection)
        #plt.title(tit)
        plt.xlabel('z (mm)')
        plt.ylabel('Interpreter')
    return plt.show()   

def p2d(V,x,y): 
    """Fits a 2D polynomial to the data in V, a 2D array of potentials.
    x and y are 2d Coordinate matricies.
    
    Returns Af, Bf, and theta; the curvatures of the Xr axis, Yr axes, and angle between X and Xr.
    We are not sure if the angle is correct when x and y are not centered on zero."""
    def s(a,N):
        """Shortcut function to convert array x into a coluumn vector."""
        import numpy as np
        a=np.reshape(a,(1,N**2)).T
        return a
  
    import numpy as np
    N=V.shape[1]
    con=np.ones((x.shape[0],x.shape[1])) # constant terms
    # Original code.
    xx,yy,xy=x*x,y*y,x*y
    xxx,yyy,xxy,xyy=xx*x,yy*y,xx*y,x*yy
    xxxx,yyyy,xxxy,xxyy,xyyy=xx*xx,yy*yy,xxx*y,xx*yy,x*yyy
    V2=s(V,N)    
    lst=[yyy,xxxy,xxyy,xyyy,xxx,yyy,xxy,xyy,xx,yy,xy,x,y,con]
    Q=s(xxxx,N)
    count = 0
    for elem in lst:
        elem=s(elem,N)
        count+=1
        Q=np.hstack((Q,elem))
    c=np.linalg.lstsq(Q,V2) # leastsq is teh closest possible in numpy
    c=c[0]
    theta=-0.5*np.arctan(c[11]/(c[10]-c[9]))
    Af=0.5*(c[9]*(1+1./np.cos(2*theta))+c[10]*(1-1./np.cos(2*theta)))
    Bf=0.5*(c[9]*(1-1./np.cos(2*theta))+c[10]*(1+1./np.cos(2*theta)))
    theta=180.*theta/np.pi
    return (Af, Bf, theta)

def plotN(trap,convention=None): # Possible to add in different conventions later.
    """Mesh the values of the DC voltage corresponding to the N DC electrodes of a planar trap,
    in a geometrically "correct" way.
    trap is a vector of N elements.
    Nikos, July 2009
    William Python 2014"""
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm 
    import mpl_toolkits.mplot3d.axes3d as p3
    N=trap.shape[0]
    n=np.floor(N/2)
    A=np.zeros((10*n,12))
    for i in range(int(n)): # Left electrodes.
        A[10*(n-i-1)+1:10*(n-i),1:3]=trap[i]
    A[:,5:7]=trap[N-1] # Central electrode.
    for i in range(1,int(n+1)): # Right electrodes.
        A[10*(n-i)+1:10*(n+1-i),9:11]=trap[i+n-1]
    x = np.arange(A.shape[0])
    y = np.arange(A.shape[1])
    x, y = np.meshgrid(x, y)
    z = A[x,y]
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.hot, linewidth=0)
    fig.colorbar(surf)
    return plt.show()

def spherharmxp(V,X,Y,Z,Xc,Yc,Zc,Order):
    """V is a 3D matrix containing an electric potential and must solve Laplace's equation
    X,Y,Z are the vectors that define the grid in three directions
     
    Xc,Yc,Zc are the coordinates of the center of the multipoles. (Specifically their values? 3/10/14)
    Order is the order of the expansion.
   
    Previously, this function expands the potential V in spherical harmonics, carried out to order Order
    i.e.: V=C00*Y00+C10*Y10+C11c*Y11c+C11s*Y11s+...
    There, the Ynm were chosen to be real, and subscript c corresponded to cos(m*phi) dependence,
    while s was sin(m*phi). 
     
    The fuction now returns coefficients in order: [C00,C1-1,C10,C11,C2-2,C2-1,C20,C21,C22,etc.] 
    This may better fit the paper, as we do not split the coefficients up by sin and cos parts of each term.
     
    The indices in V are V(i,j,k)<-> V(x,y,z).
    Nikos January 2009
    William Python Jan 2014"""
    import math as mt
    import numpy as np
    from scipy.special import sph_harm
    # Determine dimensions of V.
    s=V.shape
    nx,ny,nz=s[0],s[1],s[2] 
    # Construct variables from axes.
    [x,y,z] = np.meshgrid(X-Xc,Y-Yc,Z-Zc) # Matlab was order [y,x,z]; changed 10/19/13.
    x,y,z=np.ravel(x),np.ravel(y),np.ravel(z) # Replaced reshape, repeated for other functions.
    r,rt=np.sqrt(x*x+y*y+z*z),np.sqrt(x*x+y*y)
    # Normalize with geometric mean, 3/15/14 (most recently); makes error go down about order of magnitude
    rsort=np.sort(r)
    rmin=rsort[1] # first element is 0 
    rmax=rsort[len(r)-1] 
    rnorm=np.sqrt(rmax*rmin)
    r=r/rnorm
    # Construct theta and phi
    theta,phi=[],[] 
    for i in range(len(z)): #Set theta and phi to be correct. 10/19/13
        phi.append(mt.atan2(rt[i],z[i]))
        theta.append(mt.atan2(y[i],x[i]))
    # Make the spherical harmonic matrix in sequence of [Y00,Y-11,Y01,Y11,Y-22,Y-12,Y02,Y12,Y22...]
    # In other words: Calculate the basis vectors of the sph. harm. expansion: 
    N=nx*ny*nz
    W=np.ravel(V) # 1D array of all potential values
    W=np.array([W]).T # make into column array
    # We use the same notation as the paper, except with l in place of n.
    Yj=np.arange((len(W))) # Make a temporary first row to allow np.vstack to work. This will become the matrix.
    i=1j # Define an imaginary number.
    for n in range(Order+1):
        print('ORDER '+str(n))
        for m in range(-n,n+1):
            Y_plus=sph_harm(m,n,theta,phi)
            Y_minus=sph_harm(-m,n,theta,phi) 
            #Real conversion according to Wikipedia: Ends with same values as paper. 
            if m>0:
                yj=np.array([(1/(2**(1/2)))*(Y_plus+(-1)**m*Y_minus)])
            elif m==0:
                yj=np.array([sph_harm(m,n,theta,phi)])
            elif m<0:
                yj=np.array([(1/i*(2**(1/2)))*(Y_minus-(-1)**m*Y_plus)])
            yj=r**n*yj
            Yj=np.vstack((Yj,yj))
    Yj=np.delete(Yj,0,0) # Eliminate the termporary first row.
    # Convert the 3D DC potential into 1D array.
    # Numerically invert, here the actual expansion takes place and we obtain the expansion coefficients M_{ji}.
    Yj=Yj.T
    Mj=np.linalg.lstsq(Yj,W)
    Mj=Mj[0] # array of coefficients
    return Mj

def spherharmcmp(C,Xc,Yc,Zc,Xe,Ye,Ze,Order): 
    """This function computes the potential V from the spherical harmonic coefficients,
    which used to be: V=C1*Y00+C2*Y10+C3*Y11c+C4*Y11s+...
    There the Ynm are chosen to be real, and subscript c corresponds to cos(m*phi) dependence,
    while s is sin(m*phi).
    
    Now it is: V=C1*Y00+C2*Y1-1+C3*Y10+C4*Y11+...
    
    The expansion is carried up to multipoles of order Order.
    If the size of the coefficient vector C is not Order**2, a warning message is displayed.
    The indices in V are V(i,j,k)<-> V(x,y,z). 
    C = [C1,C2,...].T  the column vector of coefficients.
    Xc,Yc,Zc:          the coordinates of the center of the multipoles.
    Order:             the order of the expansion.
    Xe,Ye,Ze:          the vectors that define the grid in three directions.
    The input coefficients are given in the order:[C00,C1-1,C10,C11].T
    These correspond to the multipoles in cartesian coordinares:
    [c z -x -y (z^2-x^2/2-y^2/2) -3zx -3yz 3x^2-3y^2 6xy]
     1 2  3  4       5             6    7     8       9
    Nikos June 2009"""
    import numpy as np
    import math as mt 
    from scipy.special import sph_harm
    V=[]
    if C.shape[0]!=(Order+1)**2:
        while True:
            st=raw_input('spherharrmcmp.py warning:\nSize of coefficient vector not equal to Order**2. Proceed? (y/n)\n')
            if st=='n':
                return
            elif st=='y':
                break
    [x,y,z] = np.meshgrid(Xe-Xc,Ye-Yc,Ze-Zc) # order changes from y,x,z 3/9/14
    s=x.shape
    nx,ny,nz=s[0],s[1],s[2]
    x,y,z=np.ravel(x),np.ravel(y),np.ravel(z) # Replaced reshape, repeat for other functions.
    r,rt=np.sqrt(x*x+y*y+z*z),np.sqrt(x*x+y*y)
    # Normalize with geometric mean, 3/15/14 (most recently); makes error go down about order of magnitude
    rsort=np.sort(r)
    rmin=rsort[1] # first element is 0 
    rmax=rsort[len(r)-1] 
    rnorm=np.sqrt(rmax*rmin)
    r=r/rnorm
    theta,phi=[],[] 
    for i in range(len(z)): #Set theta and phi to be correct. 10/19/13
        phi.append(mt.atan2(rt[i],z[i]))
        theta.append(mt.atan2(y[i],x[i]))
    # Make the spherical harmonic matrix in sequence of [Y00,Y1-1,Y10,Y11,Y2-2,Y2-1,Y20,Y21,Y22...]
    # In other words: Calculate the basis vectors of the sph. harm. expansion:
    N=nx*ny*nz
    Yj=np.arange((len(theta))) # Make a temporary first row to allow vstack to work.
    i=1j # Define the imaginary number.
    for n in range(Order+1):
        print('ORDER '+str(n))
        for m in range(-n,n+1):
            Y_plus=sph_harm(m,n,theta,phi)
            Y_minus=sph_harm(-m,n,theta,phi) 
            #Real conversion according to Wikipedia: Ends with same values as paper. 
            if m>0:
                yj=np.array([(1/(2**(1/2)))*(Y_plus+(-1)**m*Y_minus)])
            elif m==0:
                yj=np.array([sph_harm(m,n,theta,phi)])
            elif m<0:
                yj=np.array([(1/i*(2**(1/2)))*(Y_minus-(-1)**m*Y_plus)])
            yj=r**n*yj
            Yj=np.vstack((Yj,yj))
    Yj=np.delete(Yj,0,0) # Eliminate the termporary first row.
    Yj=Yj.T 
    W=np.dot(Yj,C)
    V=W.reshape(nx,ny,nz,order='C').copy()
    return V  

def spherharmq(V,C,Xc,Yc,Zc,Order,Xe,Ye,Ze,tit):
    """This function determines the "quality" of the expansion of potential V in spherical harmonics
    It usd to be: (V=C00*Y00+C10*Y10+C11c*Y11c+C11s*Y11s+... )
    there the Ynm are chosen to be real, and subscript c corresponds to
    cos(m*phi) dependence, while s is sin(m*phi). 
    
    Now it is: (V=C00*Y00+C1-1*Y1-1+C10*Y10+C11*Y11+... )
    
    The expansion is carried up to multipoles of order Order.
    The indices in V are V(i,j,k)<-> V(x,y,z).
    
    V is the expanded potential.
    C is the coefficient vector.
    Xc,Yc,Zc are the coordinates of the center of the multipoles.
    Order is the order of the expansion.
    Xe,Ye,Ze are the vectors that define the grid in three directions.
    tit is a string describing the input potential for plot purposes. ('RF', 'DC', etc.).
    If title=='noplots' no plots are made.
    The coefficients are in the order:[C00,C1-1,C10,C11].T
    These correspond to the multipoles in cartesian coordinares:
    [c z -x -y (z^2-x^2/2-y^2/2) -3zx -3yz 3x^2-3y^2 6xy]
     1 2  3  4       5             6    7     8       9
    Nikos January 2009
    William Python Jan 2014"""
    import numpy as np
    import matplotlib.pyplot as plt
    s=V.shape
    nx,ny,nz=s[0],s[1],s[2]
    Vfit = spherharmcmp(C,Xc,Yc,Zc,Xe,Ye,Ze,Order) 
    
    # subtract lowest from each and then normalize
    Vblock = np.ones((nx,ny,nz))
    Vfit = Vfit-Vblock*np.amin(Vfit)
    Vfit = Vfit/float(np.amax(Vfit))
    V = V-Vblock*np.amin(V)
    V = V/float(np.amax(V))
    
    dV = np.subtract(V,Vfit) 
    e = np.reshape(dV,(1,nx*ny*nz))
    e=abs(e)
    f_0=np.amax(e)
    f_1=np.mean(e)
    f_2=np.median(e)
    f = np.array([f_0,f_1,f_2])
    if tit=='noplots':
        return f
    plt.plot(e[0])
    plt.title(tit)
    plt.show() 
    return f

def sumofefield(r,V,X,Y,Z,exactsaddle=True):
    """V is a 3D matrix containing an electric potential and must solve Laplace's equation
    X,Y,Z are the vectors that define the grid in three directions
    r: center position for the spherical harmonic expansion
    Finds the weight of high order multipole terms compared to the weight of
    second order multipole terms in matrix V, when the center of the multipoles
    is at x0,y0,z0.
    Used by exactsaddle for 3-d saddle search.
    Note that order of outputs for spherharmxp are changed, but 1 to 3 should still be E field.
    """
    import numpy as np
    x0,y0,z0=r[0],r[1],r[2]
    from all_functions import spherharmxp
    c=spherharmxp(V,X,Y,Z,x0,y0,z0,3) #Update these variables by abstraction.
    print ('Checking saddle: ({0},{1},{2})'.format(x0,y0,z0))
    s=c**2
    f=sum(s[1:3])/sum(s[4:9])
    print c
    real_f=np.real(f[0])
    print 'Guess:',real_f
    return real_f

def trapdepth(V,X,Y,Z,Im,Jm,Km,debug=False): 
    """Find the trap depth for trap potential V.
    Returns D,x,y,z.
    The trapping position is the absolute minimum in the potential function.
    The escape position is the nearest local maximum to the trapping position.
    D is the trap depth. This is the distance between the trapping and escape position.
        It is calculated along the vertical (X) direction
    x,y,z are the coordinates of the escape position.
    
    V is a cubic matrix of potential values
    X,Y,Z are vectors defining the grid in X,Y,Z directions.
    Im,Jm,Km are the indices of the trap potential minimum (ion position)."""  
    # Helper functions
    def a(a,N):
        """Shortcut function to convert array x into a row vector.""" 
        import numpy as np
        #a=np.reshape(a,[1,N])
        a=np.ravel(a, order='F') # Same order
        return a
    
    def index_sort(y,x):
        """Takes in two lists of the same length and returns y sorted by the indexing of x sorted."""
        xs=np.sort(x)
        ix=np.argsort(x)
        ys=np.ones(len(y)) #Sorted by the sorting defined by f being sorted. 
        for i in range(len(y)):
            j=ix[i]
            ys[i]=y[j]
        return ys
    
    import numpy as np
    import matplotlib.pyplot as plt
    if len(V.shape)!=3:
        return('Problem with findsaddle.py dimensionalities.\n')
    N1,N2,N3=V.shape
    N=N1*N2*N3
    f=V
    [Ex,Ey,Ez]=np.gradient(f) 
    E=np.sqrt(Ex**2+Ey**2+Ez**2)
    fs,Es=a(f,N),a(E,N) # Convert 3D to 1D array
    fs,Es=np.real(fs),np.real(Es)
    if debug: # plot sortings of potential and electric field to view escape position
        plt.plot(np.sort(fs)) 
        plt.title('sorted potential field')
        plt.show()
        plt.plot(np.sort(Es)) 
        plt.title('sorted electric field')
        plt.show()
        q1=index_sort(fs,Es) 
        plt.title('potential field sorted by sorted indexing of electric field')
        plt.plot(q1)
        plt.show()
        q2=index_sort(Es,fs) 
        plt.title('electric field sorted by sorted indexing of potential field')
        plt.plot(q2)
        plt.show()
    # identify the escape position and height by checking each point
    minElectricField=max(fs) # initialize as maximum E field magnitude
    distance=0
    escapeHeight=1
    escapePosition=[0,0,0]
    for i in range(N1):
        for j in range(N2):
            for k in range(N3):
                if [i,j,k]==[Im,Jm,Km]:
                    Vm=V[i,j,k]
                elif E[i,j,k]<minElectricField:
                    minElectricField=E[i,j,k]
                    escapeHeight=V[i,j,k]
                    escapePosition=[i,j,k]
                    distance=abs(Im+Jm+Km-i-j-k)          
    check=1   
    
    print minElectricField
    print escapeHeight
    print Vm
    print escapePosition
     
    if debug: 
        check=float(raw_input('How many indices away must the escape point be?'))   
    if distance<check:
        print('trapdepth.py:\nEscape point too close to trap minimum.\nImprove grid resolution or extend grid.\n')
    if escapeHeight>0.2:
        print('trapdepth.py:\nEscape point parameter too high.\nImprove grid resolution or extend grid.\n')
    D=escapeHeight-Vm
    [Ie,Je,Ke]=escapePosition
    [Xe,Ye,Ze]=[X[Ie],Y[Je],Z[Ke]]            
    return [D,Xe,Ye,Ze]