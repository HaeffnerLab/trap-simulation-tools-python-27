def test_cube():
    """Build a data struction, like the output by get_trap, to test the following functions.
    Consider a set of electrodes which produce a potential field in a cube, 10x10x10 for speed.
    Each field is built up by a linear combination of functions defined by the terms weighted by Ex,Ey,Ez,U1,U2,U3,U4,U5.
    For now, there are no constant terms and the weighting of each is binary (0 or 1).
    We will have 14 electrodes total: 3 from the 3 E's, 10 from (5 choose 2) of the U's, and one for the RF (U1 only).
    The grid for each axis varies z from -5 to 5 and the test ion is at the center.
    We will also choose trap configuration variables as appropriate and save the data in the same manner."""
    
    import numpy as np
    import datetime
    from treedict import TreeDict
    savePath = 'C:\\Python27\\trap_simulation_software\\data\\' # directory to save data at
    name = 'tk_test2'    # name to save trapping field as 
    save = True
    nue = 22
    n = 11
    Ex,Ey,Ez = 1,1,1
    U1,U2,U3,U4,U5 = 1,1,1,1,1 # x**2-y**2,2*z**2-x**2-y**2,x*y,z*y,x*z (matlab/experiment)
    trap = TreeDict()
    p=trap.potentials
    p.X,p.Y,p.Z = np.arange(-5,6),np.arange(-5,6),np.arange(-5,6) # construct grid vectors
    
    I,J,K = np.arange(-5,6),np.arange(-5,6),np.arange(-5,6) # construct background for functions
    [i,j,k] = np.meshgrid(I,J,K)
    # apply small perturbation to make saddle only pick out zeroes instead of any x=y
    #i,j,k = i-3.3*np.ones((n,n,n)),j-0.01*np.ones((n,n,n)),k-0.01*np.ones((n,n,n))
    u1,u2,u3,u4,u5 = i**2-j**2,2*k**2-i**2-j**2,i*j,k*j,i*k
    
    # Change to python/math mapping from ion trap mapping
    U1,U2,U3,U4,U5 = U3,U4,U2,U5,U1
    u1,u2,u3,u4,u5 = u3,u4,u2,u5,u1
    
    # Weird mapping?
#     Ex,Ey,Ez = Ey,Ez,Ex
#     i,j,k = j,k,i
    
    p.EL_DC_1 = Ex*i
    p.EL_DC_2 = Ey*j
    p.EL_DC_3 = Ez*k
    p.EL_DC_4 = U1*u1+U2*u2
    p.EL_DC_5 = U1*u1+U3*u3
    p.EL_DC_6 = U1*u1+U4*u4
    p.EL_DC_7 = U1*u1+U5*u5
    p.EL_DC_8 = U2*u2+U3*u3
    p.EL_DC_9 = U2*u2+U4*u4
    p.EL_DC_10= U2*u2+U5*u5
    p.EL_DC_11= U3*u3+U4*u4
    p.EL_DC_12= U3*u3+U5*u5
    p.EL_DC_13= U4*u4+U5*u5
    p.EL_DC_14= np.zeros((n,n,n))+u1
    p.EL_DC_15= np.zeros((n,n,n))+u2
    p.EL_DC_16= np.zeros((n,n,n))+u3
    p.EL_DC_17= np.zeros((n,n,n))+u4
    p.EL_DC_18= np.zeros((n,n,n))+u5
    p.EL_DC_19= np.zeros((n,n,n))
    p.EL_DC_20= np.zeros((n,n,n))
    p.EL_DC_21= np.zeros((n,n,n))
    p.EL_DC_22= u3
    p.EL_RF= p.EL_DC_22
    
    c=trap.configuration
    c.rfBias=False
    c.charge=1.60217646e-19
    c.mass=1.67262158e-27
    c.numUsedElectrodes = nue
    c.electrodeMapping = np.array([[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7],[8,8],[9,9],[10,10],
                                   [11,11],[12,12],[13,13],[14,14],[15,15],[16,16],[17,17],[18,18],
                                   [19,19],[20,20],[21,21],[22,22]])                         
    c.manualElectrodes = np.zeros(nue)
    c.usedMultipoles = [1,1,1,1,1,1,1,1]
    c.position = 0 
    c.dataPointsPerAxis = np.shape(p.Z)
    c.date = datetime.datetime.now().date()
    
    trap.potentials = p
    trap.configuration = c
    
    if save:
        import pickle
        name=savePath+name+'.pkl'
        print ('Saving '+name+' as a data structure...')
        output = open(name,'wb')
        pickle.dump(trap,output)
        output.close()
        
        
    from all_functions import plot_potential,find_saddle,exact_saddle,set_dc,dc_potential,spher_harm_exp
    from project_parameters import dcVoltages, manualElectrodes,weightElectrodes
    #print plotpot(p.EL_DC_22,p.X,p.Y,p.Z,'1D plots','title','ylab',[0,0,0])
    V=p.EL_DC_4
    print np.real(spher_harm_exp(V,0,0,0,p.X,p.Y,p.Z,2))
     
    #VMULT= set_dc()
    VMULT= dcVoltages       #all 1
    VMAN = manualElectrodes #all 0
    IMAN = weightElectrodes #all 0
    Vtest = dc_potential('C:\\Python27\\trap_simulation_software\\data\\temporary4.pkl',VMULT,VMAN,IMAN,[Ex,Ey,Ez],save)
    #print plot_potential(Vtest,p.X,p.Y,p.Z,'1D plots','title','ylab',[0,0,0])
     
    return
