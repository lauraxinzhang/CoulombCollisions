import numpy as np

KLSEP = 20 # separation between the lowest period of Fourier series and dt

class KarhunenLoeve():
    '''
    Each instantiation of the class is a determined Wiener sample path
    '''
    def __init__(self, T, seeded = False, seed = 0, N = 500):
        '''
        T: interval of time (cannot change, used in normalization)
        seeded: set to True to manually seed the Wiener process
        seed:   when seeded = True, random generator is seeded with this value
        N: highest order of Fourier Series
        '''
        
        self.N = N
        self.T = T
        self.seeded = seeded
        self.seed = seed
        self.Zn = None
        
        self.resetZn()
        return
    
    def resetZn(self):
        '''
        Resets the Fourier coefficients.
        note: only called when N is changed.
        '''
        if self.seeded:
            np.random.seed(self.seed)
        mean = np.array([0, 0, 0])
        cov = np.eye(3)
        if np.sum(self.Zn) != None: # if this is not being called by the constructor
            print("Warning: Zn has been reset.")
        self.Zn = np.random.multivariate_normal(mean, cov, size = (self.N)) # N by 3        
        return
    
    def resetN(self, N):
        self.N = N
        self.resetZn()
        return
    
    def phi(self, t):
        '''
        Calculates the Fourier component n
        t: input list of time
        '''
        nList = np.linspace(0, self.N-1, self.N)
        factor = 2*np.sqrt(2 * self.T)/((2*nList + 1)*np.pi)
        arg = (2*nList[:, None] + 1)*np.pi*t[None, :]/(2 * self.T)
        return factor[:, None] * np.sin(arg) # N by size(t); N number of fourier series
    
    def period(self):
        '''
        return the period of the highest frequency component
        '''
#         return 2 * np.pi * 2 * self.T /((2*self.N + 1)*np.pi)
        return 4 * self.T / (2 * self.N + 1)
    
    def order(self, pr):
        '''
        calculate the order of Fourier components needed for a given time interval
        pr: requested period for the highest frequency
        '''
        N = (2*self.T / pr) + 0.5 # add 0.5 so that (int) cast gives the ceiling instead of floor.
        return int(N)
        
    
    def W(self, t):
        '''
        t: list of input points
        seeded: whether to seed the generator to fix the random state
        seed:   seed of random number generator, used when seeded = True
        '''
        dtList = np.diff(t)
        dtmin = np.min(dtList)
        dtLimit = self.period()
        if dtmin < dtLimit * KLSEP :
            period_r = dtmin / KLSEP # request a new period that's 50 times the minimum dt
            self.resetN(self.order(period_r))
            print("order of Fourier series adjusted to {}.".format(self.N))
            
        phi = self.phi(t) # N by size(t)
        Zn = self.Zn # N by 3
        product = Zn[:, None, :] * phi[:, :, None] # N x size(t) x 3
        return np.sum(product, axis = 0) # sum over N, size(t) x 3