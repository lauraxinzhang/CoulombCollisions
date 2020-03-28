import numpy as np
import sys
from scipy import special
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime

ME=9.10938356E-31                       # electron mass, in kilogram
MI=1.6726219E-27                 # ion mass, in kilogram
QE=1.60217662E-19                # elementary charge, in coulomb
EPSILON0=8.8541878128E-12

LOGLAMBDA = 19

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
        self.wiener = None
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
    
    def dv_diff_EM(self, dt, v, dW):
        '''
        returns the diffusion dv vector from Euler-Maruyama
        '''
        
        bs = v.shape[0] #batch size of velocities

        nupara = self.nu_par(v)
        nuperp = self.nu_perp(v)
        speed = np.linalg.norm(v, axis = -1)
        Dpara_sqrt  = speed * np.sqrt(0.5 * nupara) * self.para
        Dperp_sqrt = speed * np.sqrt(0.5 * nuperp) * self.perp
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
    
    def dv_diff_MEM(self, dt, v, dW):
        '''
        returns the updated velocity from combined diffusive processes
        '''
        bs = v.shape[0]
        
        nupara = self.nu_par(v)
        nuperp = self.nu_perp(v)
        
        speed = np.linalg.norm(v, axis = -1)
        Dpara_sqrt  = speed * np.sqrt(0.5 * nupara) * self.para
        Dperp_sqrt = speed * np.sqrt(0.5 * nuperp) * self.perp
        
        diag = np.identity(3)
        identities = np.tile(diag, (bs, 1)).reshape(bs, 3, 3)
        
        omega = np.cross(v, dW)
        omega_hat = np.apply_along_axis(self.hatMap, 1, omega)
        
        T_const = 0.5 * (Dpara_sqrt - Dperp_sqrt)* (speed**-2)

        T_n = T_const[:, None, None] * omega_hat
        A = np.linalg.inv(identities - T_n)
        B = Dpara_sqrt[:, None] * dW
        C = np.squeeze(np.matmul((identities + T_n), np.expand_dims(v, axis = -1)))
        RHS = np.expand_dims(B + C, axis = -1)
        result = np.squeeze(np.matmul(A, RHS))
        return result
    
    def fixedWiener(self, nparts, tTot):
        '''
        Generate a fixed fine-grain Wiener process, pass to class members
        Note: This gauranteed that for each class instantiation, the 'fixed' underlying
                Wiener process is defined only one time.
        '''
#         np.random.seed(0)
        if self.dWt == None:
            self.dWt = tTot * 1E-5
            length = int(tTot/self.dWt) *2
            mean = [0, 0, 0]
            var = np.eye(3) * self.dWt
            self.wiener = np.random.multivariate_normal(mean, var, (length,nparts))
#             print(self.wiener.shape)
        return
        
    
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
            iNow  = int(tNow / self.dWt)
            iNext = int(tNext/ self.dWt)
                
            try:
                delta_W = self.wiener[iNext, :,:] - self.wiener[iNow, :,:] # (nparts x 3)
#                 return np.tile(delta_W, (bs, 1)) # return the same wiener process for each sample path
                return delta_W
            
            except IndexError:
                print("Error: Trying to reach beyond prepared underlying Wiener process. Adjust Ttot.")
                sys.exit(1)
                

            
    
    def beam_EM(self,E, nparts, dt, tTot, fixedWiener = False):
        '''
        E:      energy of beam particles [eV]
        nparts: number of particles [#]
        tTot:   total time of simulation [s]
        return: History of velocities (timesteps * nparts * 3)
        '''
        
        
        if fixedWiener:
            self.fixedWiener(nparts, tTot)
            #print('fixed wiener', series[1, :])
#         else: 
           # np.random.seed(0) # not a true random
            
        v_beam = np.sqrt(2*E*QE/self.ma)
        v = np.tile(np.array([v_beam, 0, 0]), (nparts, 1)) # initialize nparts particles with same velocity
        tnow = 0
        vHist = np.array([v])
        while tnow < tTot:
            dW = self.wienerProcess(dt, nparts, fixedWiener, tnow)
            #print(dW[:3, :])
            dv_s = self.dv_slow(dt, v)
            dv_diff = self.dv_diff_EM(dt, v, dW)
            v = v + dv_s + dv_diff
            vHist = np.append(vHist,np.array([v]), axis = 0)
            tnow += dt
        return vHist
    
    def beam_MEM(self,E, nparts, dt, tTot, fixedWiener = False):
        '''
        E:      energy of beam particles [eV]
        nparts: number of particles [#]
        tTot:   total time of simulation [s]
        return: History of velocities (timesteps * nparts * 3)
        '''
        if fixedWiener:
            self.fixedWiener(nparts, tTot)
            #print('fixed wiener')
#         else: 
           # np.random.seed(0) # not a true random
        
        v_beam = np.sqrt(2*E*QE/self.ma)
        v = np.tile(np.array([v_beam, 0, 0]), (nparts, 1)) # initialize nparts particles with same velocity
        tnow = 0
        vHist = np.array([v])
        while tnow < tTot:
            dW = self.wienerProcess(dt, nparts, fixedWiener, tnow)
            #if fixedWiener: print(dW)
            dv_s = self.dv_slow(dt, v)
            v = self.dv_diff_MEM(dt, v, dW) + dv_s
            vHist = np.append(vHist,np.array([v]), axis = 0)
            tnow += dt
        return vHist
    
    
    def energyConvergence(self, E, nparts, dt_start, dt_end, t_tot, numTrials, 
                          integrator = 'EM', fixedWiener = False, silent = False):
        '''
        Integrates the system over the same length of time, at different dt
        
        nparts:     number of particles per run
        dt_start:   the lowest order of magnitude for dt [log(dt)]
        dt_end:     the highest order of magnitude
        t_tot:      Duration of each simulation
        numTrials:  Number of trials per dt
        integrator: Method for integrating diffusion, 'EM' or 'MEM'
        '''
        order = dt_start
        dtList = np.array([])
        meanList = np.array([])
        varList = np.array([])
        while order <= dt_end:
            dt = 10**order
            if not silent: print(dt)
            i = 0
            while i < numTrials:
                if fixedWiener:
                    self.dWt = None # flag a reset of the underlying Wiener
                    np.random.seed(i) # use the same states of random numbers for each dt
                    
                if integrator == 'EM':
                    vHist = self.beam_EM(E, nparts, dt, t_tot, fixedWiener)
                elif integrator == 'MEM':
                    vHist = self.beam_MEM(E, nparts, dt, t_tot, fixedWiener)
                else:
                    print('Integrator option not recognized: ', integrator)
                    return 0, 0, 0, 0
                if not silent: print('done with trial', i)
                speedHist = np.linalg.norm(vHist, axis = -1)**2
                dtList = np.append(dtList, dt)
                final = speedHist[len(speedHist)-1,:]
                meanList = np.append(meanList, np.mean(final))
                varList = np.append(varList, np.var(final))
                i+=1
            order+=0.2
        return dtList, meanList, varList, speedHist[0, 0]
    
    def pitchConvergence(self, E, nparts, dt_start, dt_end, t_tot, numTrials, 
                         integrator = 'EM', fixedWiener = False, silent = False):
        '''
        nparts:     number of particles per run
        dt_start:   the lowest order of magnitude for dt [log(dt)]
        dt_end:     the highest order of magnitude
        t_tot:      Duration of each simulation
        numTrials:  Number of trials per dt
        integrator: Method for integrating diffusion, 'EM' or 'MEM'
        '''
        order = dt_start
        dtList = np.array([0])
        meanList = np.array([0, 0, 0])
        varList = np.array([0, 0, 0])
        pitch_meanList = np.array([0])
        pitch_varList = np.array([0])
        while order <= dt_end:
            dt = 10**order
            if not silent: print(dt)
            i = 0
            while i < numTrials:
                if fixedWiener:
                    self.dWt = None # flag a reset of the underlying Wiener
                    np.random.seed(i) # use the same states of random numbers for each dt
                    
                if integrator == 'EM':
                    vHist = self.beam_EM(E, nparts, dt, t_tot, fixedWiener)
                elif integrator == 'MEM':
                    vHist = self.beam_MEM(E, nparts, dt, t_tot, fixedWiener)
                else:
                    print('Integrator option not recognized: ', integrator)
                    return 0, 0, 0, 0, 0, 0
                if not silent: print('done with trial', i)
                
                dtList = np.append(dtList, dt)
                
                v_mean_final = np.mean(vHist[-1, :, :], axis = 0) # average over particles (3)
                v_var_final = np.var(vHist[-1, :, :], axis = 0) # average over particles (3)

                meanList = np.vstack((meanList, v_mean_final))
                varList = np.vstack((varList, v_var_final))
                
                v_perp = np.sqrt(vHist[-1, :, 1]**2 + vHist[-1, :, 2]**2)
                
                pitch_mean = np.mean(v_perp / vHist[-1, :, 0], axis = 0)
                pitch_var = np.var(v_perp / vHist[-1, :, 0], axis = 0)
                
                pitch_meanList = np.vstack((pitch_meanList, pitch_mean))
                pitch_varList = np.vstack((pitch_varList, pitch_var))
                i+=1
            order+=0.2
        return dtList, meanList, varList, pitch_meanList, pitch_varList, vHist[0, 0, :]

    
    def pitchIter(self, E, nparts, dt, i_start, i_end, di, numTrials, 
                  integrator = 'EM', fixedWiener = False, silent = False):
        '''
        nparts:     number of particles per run
        dt:         time step [s]
        i_start:    starting number of iterations
        i_end:      end number of iterations
        di:         step size in iterations
        numTrials:  Number of trials per i
        integrator: Method for integrating diffusion, 'EM' or 'MEM'
        '''
        iterNow = i_start
        iList = np.array([0])
        meanList = np.array([0, 0, 0])
        varList = np.array([0, 0, 0])
        pitch_meanList = np.array([0])
        pitch_varList = np.array([0])
        while iterNow <= i_end:
            t_tot = dt * iterNow
            if not silent: print('Pushing ',iterNow, 'iterations, for ', t_tot, 'seconds...')
            i = 0
            while i < numTrials:
                if fixedWiener:
                    self.dWt = None # flag a reset of the underlying Wiener
                    np.random.seed(i) # use the same states of random numbers for each dt
                    
                if integrator == 'EM':
                    vHist = self.beam_EM(E, nparts, dt, t_tot, fixedWiener)
                elif integrator == 'MEM':
                    vHist = self.beam_MEM(E, nparts, dt, t_tot, fixedWiener)
                else:
                    print('Integrator option not recognized: ', integrator)
                    return 0, 0, 0, 0, 0, 0
                if not silent: print('done with trial', i)
                
                iList = np.append(iList, iterNow)
                
                v_mean_final = np.mean(vHist[-1, :, :], axis = 0) # average over particles (3)
                v_var_final = np.var(vHist[-1, :, :], axis = 0) # average over particles (3)

                meanList = np.vstack((meanList, v_mean_final))
                varList = np.vstack((varList, v_var_final))
                
                v_perp = np.sqrt(vHist[-1, :, 1]**2 + vHist[-1, :, 2]**2)
                
                pitch_mean = np.mean(v_perp / vHist[-1, :, 0], axis = 0)
                pitch_var = np.var(v_perp / vHist[-1, :, 0], axis = 0)
                
                pitch_meanList = np.vstack((pitch_meanList, pitch_mean))
                pitch_varList = np.vstack((pitch_varList, pitch_var))
                i+=1
                
            iterNow += di
        return iList[1:], meanList[1:, :], varList[1:, :], pitch_meanList[1:], pitch_varList[1:], vHist[0, 0, :]