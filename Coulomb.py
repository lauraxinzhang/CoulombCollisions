import numpy as np
import sys
from scipy import special
from scipy import stats
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
import time

from KarhunenLoeve import * # For a smooth representation of a Wiener process

ME=9.10938356E-31                       # electron mass, in kilogram
MI=1.6726219E-27                 # ion mass, in kilogram
QE=1.60217662E-19                # elementary charge, in coulomb
EPSILON0=8.8541878128E-12

LOGLAMBDA = 19
LBASE = 2

class Coulomb():
   
    def __init__(self, qa, qb, ma, mb, temp_b, n, slow = True, para = True, perp = True):
        '''
        qa, ma: charge and mass of test particle [kg, Coulomb]
        qb, mb: charge and mass of background particle [kg, Coulomb]
        temp_b: temperature of background species [eV]
        n:      density of background species [m^-3]
        '''
        self.qa = qa
        self.qb = qb
        self.ma = ma
        self.mb = mb
        self.temp_b = temp_b
        self.temp_j = temp_b * QE # background temperature in Joules
        self.n = n
        
        self.slow = slow
        self.para = para
        self.perp = perp
        
        self.dWt = None
        self.kl = None
        
#         self.v_next = None # a placeholder for remembering the predictor velocities for predictor - corrector
#         self.cor = None    # placeholder for MEM corrector order
        return
    
    def erf(self, x): 
        '''Error function erf(x)'''
        return special.erf(x)
    
    def nu_ab(self):
        return self.n * (self.qa**2) * (self.qb**2) * LOGLAMBDA / (4*np.pi * (EPSILON0**2) * (self.ma**2))
    
    def ratio_x(self, v):
        vt = np.sqrt(2*self.temp_j / self.mb);
        return v/vt
    
    def G(self, x):
        '''The Chanderaskar G function'''
        gauss = ((2 * x) / np.sqrt(np.pi)) * np.exp(-1 * x**2);
        return (self.erf(x) - gauss)/(2 * x**2)
    
    def nu_s(self, v):
        '''Slowing down frequency'''
        speed = np.linalg.norm(v, axis = -1)
        xb = self.ratio_x(speed);
        return (1/(speed * self.temp_j)) * self.G(xb)* (self.ma + self.mb) * self.nu_ab()
    
    def nu_par(self, v):
        speed = np.linalg.norm(v, axis = -1)
        xb = self.ratio_x(speed)
        return 2 * self.nu_ab() * self.G(xb) / (speed ** 3)
    
    def nu_perp(self, v):
        speed = np.linalg.norm(v, axis = -1)
        xb = self.ratio_x(speed)
        return self.nu_ab() * (self.erf(xb) - self.G(xb))/ (speed ** 3)
    
    def dv_slow(self, dt, v):
        '''
        returns the slowing down dv vector
        '''
        if self.slow:
            bs = v.shape[0]
            nus = self.nu_s(v).reshape(bs, 1)
            return -1 * dt * nus * v # forward Euler for deterministic slowing down
        else:
            return np.zeros(v.shape)
    
    def dv_strat(self, dt, v):
        '''
        returns the stratanovich correction to the SDE; to be used in EM method.
        '''
        nuperp = self.nu_perp(v)
        speed = np.linalg.norm(v, axis = -1)
#         mult = -0.5 * nuperp * dt
        mult = -1 * nuperp * dt

        return mult[:, None] * v
    
    def dv_diff_EM(self, dt, v, dW):
        '''
        returns the diffusion dv vector from Euler-Maruyama
        '''
        
        bs = v.shape[0] #batch size of velocities

        nupara = self.nu_par(v)
        nuperp = self.nu_perp(v)
        speed = np.linalg.norm(v, axis = -1)
#         Dpara_sqrt  = speed * np.sqrt(0.5 * nupara) * self.para
#         Dperp_sqrt = speed * np.sqrt(0.5 * nuperp) * self.perp
        Dpara_sqrt  = speed * np.sqrt(nupara) * self.para
        Dperp_sqrt = speed * np.sqrt(nuperp) * self.perp

#         Dperp_sqrt = Dperp_sqrt / Dperp_sqrt # setting diffusion coef to 1
        
        diag = np.identity(3)
        identities = np.tile(diag, (bs, 1)).reshape(bs, 3, 3)
        vvs = v[:, :, None] * v[:, None, :] / (speed**2)[:, None, None]
        D_mat = vvs * Dpara_sqrt[:, None, None] + (identities - vvs) * Dperp_sqrt[:, None, None]
        
        wiener = np.expand_dims(dW, axis = -1)

        return np.squeeze(np.matmul(D_mat, wiener), -1)
    
    def hatMap(self, vector):
        '''
        converts a given vector to it's corresponding anti-symmetric permutation matrix
        Note: for use with np.apply_along_axis as a functor
        '''
        v1 = vector[0]
        v2 = vector[1]
        v3 = vector[2]
        return np.array([[0, -1*v3, v2], [v3, 0, -1 * v1], [-1*v2, v1, 0]])
    
    def dv_diff_MEM(self, dt, v, dW, cor):
        '''
        returns the updated velocity from combined diffusive processes
        '''
        bs = v.shape[0]
        
        nupara = self.nu_par(v)
        nuperp = self.nu_perp(v)
        
        speed = np.linalg.norm(v, axis = -1)
#         Dpara_sqrt  = speed * np.sqrt(0.5 * nupara) * self.para
#         Dperp_sqrt = speed * np.sqrt(0.5 * nuperp) * self.perp
        Dpara_sqrt  = speed * np.sqrt(nupara) * self.para
        Dperp_sqrt = speed * np.sqrt(nuperp) * self.perp
        
        diag = np.identity(3)
        identities = np.tile(diag, (bs, 1)).reshape(bs, 3, 3)
        
        T_const = 0.5 * (Dpara_sqrt - Dperp_sqrt)* (speed**-2)

        v_next = v
        for order in range(cor+1):
            v_half = 0.5 * (v + v_next)
            omega = np.cross(v_half, dW)
            
            omega_hat = np.apply_along_axis(self.hatMap, 1, omega)


            T_n = T_const[:, None, None] * omega_hat
            A = np.linalg.inv(identities - T_n)
            B = Dpara_sqrt[:, None] * dW
            C = np.squeeze(np.matmul((identities + T_n), np.expand_dims(v, axis = -1)))
            RHS = np.expand_dims(B + C, axis = -1)
            v_new = np.squeeze(np.matmul(A, RHS))
#             print('{}th correction: '.format(order), v_new - v_next)
            v_next = v_new
            
        return v_next
    
    def fixedWiener(self, tTot, seed = 0.0):
        '''
        Generate a fixed fine-grain Wiener process, pass to class members
        Note: This gauranteed that for each class instantiation, the 'fixed' underlying
                Wiener process is defined only one time.
        '''
#         if self.dWt == None:
        if False: # the discrete representation is deprecated
            self.dWt = dt * 1E-4
            length = int(tTot/self.dWt) *2
            mean = [0, 0, 0]
            var = np.eye(3) * self.dWt
            self.wiener = np.random.multivariate_normal(mean, var, (length,nparts))
        
        self.kl = KarhunenLoeve(tTot, seeded = True, seed = seed)
        return

#     def KLWiener(self, dt, tTot):
#         self.kl = KarhunenLoeve(tTot)
#         return
        
        
    
    def wienerProcess(self, dt, bs, fixedWiener = False, tNow = None):
        '''
        returns a wiener process dW, same shape as the currently v inputs
        dt: time step of the simulation
        bs: batch size of the input
        [dWt, wiener]: the dWt that the fixed Wiener process with generated with, and the series itself
        tNow: Current time in simulation
        '''
        if not fixedWiener:
            #print('not fixed')
            mean = [0, 0, 0]
            var = np.eye(3) * dt #[[dt, 0, 0], [0, dt, 0], [0, 0, dt]]
            wiener = np.random.multivariate_normal(mean, var, bs)
            return wiener
        else:
            #print('fixed')
            tNext = tNow + dt
#             iNow  = int(tNow / self.dWt)
#             iNext = int(tNext/ self.dWt)
            if bs != 1:
                print("Why are you using more than 1 particle for 1 fixed sample path?? I refuse.")
                sys.exit(1)
            try:
#                 delta_W = np.sum(self.wiener[iNow : iNext + 1], axis = 0)
#                 return delta_W
                tList = np.array([tNow, tNext])
                W = self.kl.W(tList) # 2 tpoints x 3 vcomp
                return np.diff(W, axis = 0) # 1 x 3 vcomp
            
#             except IndexError:
#                 print("Error: Trying to reach beyond prepared underlying Wiener process. Adjust Ttot.")
#                 sys.exit(1)
            except AttributeError:
                print("KL series hasn't been initialized")
                sys.exit(1)
                            
    
    def beam_EM(self,E, nparts, dt, tTot, snaps, strat = True, fixedWiener = False):
        '''
        E:      energy of beam particles [eV]
        nparts: number of particles [#]
        tTot:   total time of simulation [s]
        snaps:  number of snapshots to output; set to '1' for only the initial and final states
        strat:  Whether to include the Stratanovich correction
        fixedWiener: whether to use a fixed underlying Wiener process
        return: [tHist, vHist] History of velocities (timesteps * nparts * 3) and the corresponding times
        '''
        
        
#         if fixedWiener:
#             self.fixedWiener(nparts, dt, tTot)
            
        v_beam = np.sqrt(2*E*QE/self.ma)
        v = np.tile(np.array([v_beam, 0, 0]), (nparts, 1)) # initialize nparts particles with same velocity
        tnow = 0
        
        # Some preparations so that we can modify vHist by index
        stepsTotal = int(tTot/dt)
        if stepsTotal < snaps:
            snaps = stepsTotal
        try:
            dStep = int(stepsTotal / snaps)
        except ZeroDivisionError:
            print("snaps can't be 0, setting to 1 for outputtting the final state")
            snaps = 1
            dStep = int(stepsTotal)
            
        vHist = np.zeros((int(snaps+1), nparts, 3))
        tHist = np.zeros((int(snaps+1),))
        
        for step in range(stepsTotal):

            if (dStep != 0 and step % dStep == 0):
                index = int(step / dStep)
                vHist[index] = v
                tHist[index] = tnow
                
            dW = self.wienerProcess(dt, nparts, fixedWiener, tnow)
#             print(dW[:3, :])
            dv_s = self.dv_slow(dt, v)
            dv_diff = self.dv_diff_EM(dt, v, dW)
            dv_strat = self.dv_strat(dt, v) * strat
            v = v + dv_s + dv_diff + dv_strat
            
            tnow += dt
        #Always keep the final state
        vHist[-1] = v
        tHist[-1] = tnow
        return tHist, vHist
    
    
    def beam_MEM(self,E, nparts, dt, tTot, snaps, correctors = 1, fixedWiener = False):
        '''
        E:      energy of beam particles [eV]
        nparts: number of particles [#]
        tTot:   total time of simulation [s]
        return: History of velocities (timesteps * nparts * 3)
        '''
#         if fixedWiener:
#             self.fixedWiener(nparts, dt, tTot)
#         self.cor = correctors
            
        v_beam = np.sqrt(2*E*QE/self.ma)
        v = np.tile(np.array([v_beam, 0, 0]), (nparts, 1)) # initialize nparts particles with same velocity
        tnow = 0
        
        # Some preparations so that we can modify vHist by index
        stepsTotal = int(tTot/dt)
        if stepsTotal < snaps:
            snaps = stepsTotal
        try:
            dStep = int(stepsTotal / snaps)
        except ZeroDivisionError:
            print("snaps can't be 0, setting to 1 for outputtting the final state")
            snaps = 1
            dStep = int(stepsTotal)
            
        vHist = np.zeros((int(snaps+1), nparts, 3))
        tHist = np.zeros((int(snaps+1),))
        
        for step in range(stepsTotal):
#             if correctors != 0:
#                 self.v_prev = v
                
            if (dStep != 0 and step % dStep == 0):
                index = int(step / dStep)
                vHist[index] = v
                tHist[index] = tnow
                
                
            dW = self.wienerProcess(dt, nparts, fixedWiener, tnow)
            #if fixedWiener: print(dW)
            dv_s = self.dv_slow(dt, v)
            v_diff = self.dv_diff_MEM(dt, v, dW, correctors)             
            v = v_diff + dv_s
            tnow += dt
        #Always keep the final state
        vHist[-1] = v
        tHist[-1] = tnow
        return tHist, vHist
    
    
    def energyConvergence(self, E, nparts, dt_start, dt_end, t_tot, numTrials, moments = [1, 2],
                          integrator = 'EM', silent = False):
        '''
        nparts:     number of particles per run
        dt_start:   the lowest order of magnitude for dt [log(dt)]
        dt_end:     the highest order of magnitude
        t_tot:      Duration of each simulation
        numTrials:  Number of trials per dt
        moments :   [int] Python list of central moments to calculate
        integrator: Method for integrating diffusion, 'EM' or 'MEM'
        '''
        numT = int((dt_end - dt_start) / 0.2) # number of points in dt
                
        dtList = np.linspace(dt_end, dt_start, numT)
        momList = np.zeros((numT, numTrials, len(moments), 1))
        
        for i in range(numT):
            dt = 10**dtList[i]
            if not silent: print(dt)
            trialList = np.zeros((numTrials, len(moments), 1))
            for j in range(numTrials):
                    
                if integrator == 'EM':
                    tHist, vHist = self.beam_EM(E, nparts, dt, t_tot, snaps = 2, strat = True, fixedWiener = False)
                elif integrator == 'MEM':
                    tHist, vHist = self.beam_MEM(E, nparts, dt, t_tot, snaps = 2, fixedWiener = False)
                else:
                    print('Integrator option not recognized: ', integrator)
                    return 0, 0, 0, 0
                if not silent: print('done with trial', j)
                    
                speed = np.linalg.norm(vHist[-1], axis = -1)
                momOut = np.zeros((len(moments), 1))
                for orderIndex in range(len(moments)):
                    order = moments[orderIndex]
                    if order == 1:
                        moment = np.mean(vHist[-1], axis = 0)
                    elif order == 2:
                        moment = np.var(vHist[-1], axis = 0)
                    else:
                        moment = stats.moment(vHist[-1], moment = order, axis = 0)
                    momOut[orderIndex] = moment
                trialList[j] = momOut
            
            momList[i] = trialList
            dtList[i] = dt
        return dtList, momList  
    
    def pitchConvergence(self, E, nparts, numT, t_tot, numTrials, moments = [1, 2],
                         integrator = 'EM', correctors = None,
                         silent = False):
        '''
        nparts:     number of particles per run
        dt_start:   the lowest order of magnitude for dt [log(dt)]
        dt_end:     the highest order of magnitude
        t_tot:      Duration of each simulation
        numTrials:  Number of trials per dt
        moments :   [int] Python list of central moments to calculate
        integrator: Method for integrating diffusion, 'EM' or 'MEM'
        '''
        
#         numT = int((dt_end - dt_start) / 0.2) # number of points in dt
                
#         dtList = np.linspace(dt_end, dt_start, numT)
        dtReal = np.zeros(numT)
        momList = np.zeros((numT, numTrials, len(moments), 3))
        
        for i in range(numT):
#             dt = 10**dtList[i]
            dt = t_tot / (LBASE**(i+2))
            if not silent: print(dt)
            trialList = np.zeros((numTrials, len(moments), 3))
            for j in range(numTrials):
                    
                if integrator == 'EM':
                    tHist, vHist = self.beam_EM(E, nparts, dt, t_tot, snaps = 2, strat = True, fixedWiener = False)
                elif integrator == 'MEM':
                    tHist, vHist = self.beam_MEM(E, nparts, dt, t_tot, snaps = 2, correctors = correctors, 
                                                 fixedWiener = False)
                else:
                    print('Integrator option not recognized: ', integrator)
                    return 0, 0, 0, 0
                if not silent: print('done with trial', j)
                    
#                 speed = np.linalg.norm(vHist[-1], axis = -1)
                momOut = np.zeros((len(moments), 3))
                for orderIndex in range(len(moments)):
                    order = moments[orderIndex]
                    if order == 1:
                        moment = np.mean(vHist[-1], axis = 0)
                    elif order == 2:
                        moment = np.var(vHist[-1], axis = 0)
                    else:
                        moment = stats.moment(vHist[-1], moment = order, axis = 0)
                    momOut[orderIndex] = moment
                trialList[j] = momOut
            
            momList[i] = trialList
            dtReal[i] = dt
        return dtReal, momList

    
    def pitchIter(self, E, nparts, dt, i_start, i_end, di, numTrials, moments = [1, 2],
                  integrator = 'EM', correctors = None,
                  silent = False):
        '''
        nparts:     number of particles per run
        dt:         time step [s]
        i_start:    starting number of iterations
        i_end:      end number of iterations
        di:         step size in iterations
        numTrials:  Number of trials per i
        moments :   [int] Python list of central moments to calculate
        integrator: Method for integrating diffusion, 'EM' or 'MEM'
        '''
        numI = int((i_end - i_start)/di)
        iList = np.linspace(i_start, i_end, numI)
        momList = np.zeros((numI, numTrials, len(moments), 3))
        for i in range(numI):
            t_tot = dt * iList[i]
            if not silent: print('Pushing ',iterNow, 'iterations, for ', t_tot, 'seconds...')
                
            trialList = np.zeros((numTrials, len(moments), 3))
            for j in range(numTrials):
                    
                if integrator == 'EM':
                    tHist, vHist = self.beam_EM(E, nparts, dt, t_tot, snaps = 2, strat = True, fixedWiener = False)
                elif integrator == 'MEM':
                    tHist, vHist = self.beam_MEM(E, nparts, dt, t_tot, snaps = 2, correctors = correctors, 
                                                 fixedWiener = False)
                else:
                    print('Integrator option not recognized: ', integrator)
                    return 0, 0, 0, 0
                if not silent: print('done with trial', j)
                    
                momOut = np.zeros((len(moments), 3))
                for orderIndex in range(len(moments)):
                    order = moments[orderIndex]
                    if order == 1:
                        moment = np.mean(vHist[-1], axis = 0)
                    elif order == 2:
                        moment = np.var(vHist[-1], axis = 0)
                    else:
                        moment = stats.moment(vHist[-1], moment = order, axis = 0)
                    momOut[orderIndex] = moment
                trialList[j] = momOut
            
            momList[i] = trialList
#             iList[i] = dt
        return iList, momList


    def strongConv(self, E, npaths, numT, t_tot, 
                             integrator = 'EM', correctors = None, silent = False):
        '''
        E:          Energy of beam (eV)
        npaths:     number of sample paths
        numT:       Number of times to divide t_tot by 4 (number of time steps)
        t_tot:      Duration of each simulation
        integrator: Method for integrating diffusion, 'EM' or 'MEM'
        correctors: Order of corrector for MEM
        '''
#         numT = int((dt_end - dt_start) / 0.2) # number of points in dt
                
#         dtList = np.linspace(dt_end, dt_start, numT)
#         numT = l
        dtReal = np.zeros(numT)
#         errList = np.zeros((npaths, numT - 1, 3))
        errList = np.zeros((npaths, numT, 3))
        lList = np.arange(numT)
        
        for i in range(npaths):
#             self.dWt = None # flag a reset of the underlying Wiener
            self.fixedWiener(t_tot, seed = int(time.time()) )
            trialList = np.zeros((numT, 3))
#             for j in range(numT):
            for j in np.flip(lList):
#                 dt = 10**dtList[j]
                dt = t_tot / (LBASE**(j+2))
                if not silent: print(dt)
                
                if integrator == 'EM':
                    tHist, vHist = self.beam_EM(E, 1, dt, t_tot, snaps = 2, strat = True, fixedWiener = True)
                elif integrator == 'MEM':
                    tHist, vHist = self.beam_MEM(E, 1, dt, t_tot, snaps = 2, correctors = correctors, 
                                                 fixedWiener = True)
                else:
                    print('Integrator option not recognized: ', integrator)
                    return 0, 0, 0, 0
                if not silent: print('done with trial', j)
                    
                trialList[j] = np.squeeze(vHist)[-1]
#                 print(j, trialList)
                dtReal[j] = dt
                
#             errList[i] = np.abs(np.diff(trialList, axis = 0))
            if not silent: print('done with sample path', i)
#             print(trialList)
            errList[i] = trialList
#         dtReal = dtReal[:-1]
#         dtReal = np.abs(np.diff(dtReal))
        return dtReal, errList
