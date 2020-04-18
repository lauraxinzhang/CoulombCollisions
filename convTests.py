from Coulomb import *
from plotting import *
import sys, getopt

def main(argv):    
    try:
        opts, args = getopt.getopt(argv,"hw:s:",["help","weak=","strong="])
    except getopt.GetoptError:
        helpMessage()
        sys.exit(2)
        
    for opt, arg in opts:
        if opt in ('-h', "--help"):
            helpMessage()
            sys.exit()
        elif opt in ("-w", "--weak"):
            if arg in ('EM', 'MEM'):
                method = arg
                weakTest(method, args)
        elif opt in ("-s", "--strong"):
            if arg in ('EM', 'MEM'):
                method = arg
                strongTest(method, args)

def helpMessage():
    print('convTest.py -h \n', 
          '-w [method] <nparts, numT, t_tot, numTrials, cor> \n',
          '-s [method] <nparts, numT, t_tot, cor> \n')
    return

def strongTest(method, args):
    test = Coulomb(QE, QE, MI, MI, 200, 1E19, False, False, True)
    try:
        nparts, numT, t_tot, cor = args
    except ValueError:
        print('Incorrect number of arguments. Try again.')
        helpMessage()
        sys.exit()

    t, err = test.strongConv(17000, int(nparts), int(numT), float(t_tot), 
                             method, correctors = int(cor), silent = True)
    fname = 's_'+ method
    if method == 'MEM':
        fname += '_cor_' + str(cor)
    processAndWrite(t, err, fname)
    return

def weakTest(method, args):
    test = Coulomb(QE, QE, MI, MI, 200, 1E19, False, False, True)
    try:
        nparts, numT, t_tot, numTrials, cor = args
    except ValueError:
        print('Incorrect number of arguments. Try again.')
        helpMessage()
        sys.exit()
    mom = [1, 2]
    t, momList = test.pitchConvergence(17000, int(nparts), int(numT), float(t_tot), 
                                       int(numTrials), 
                                       moments = mom, 
                                       integrator = method, 
                                       correctors = int(cor), 
                                       silent = True)
    # shape of momList: time x trials x moments x 3 comp
    fname = 'w_'+ method
    if method == 'MEM':
        fname += '_cor_' + str(cor)
        
    for i in range(len(mom)):
        out = momList[:, :, i, :] # time x trials  x 3 comp
        out = np.transpose(out, (1, 0, 2)) # trials x time x 3 comp
        outname = fname + '_mom{}_'.format(mom[i])
        processAndWrite(t, out, outname)
    return

def processAndWrite(t, err, fname):
    err_ave = np.mean(err, axis = 0)
    err_err = stats.sem(err, axis = 0)
    np.savetxt(fname + '_ave.txt', err_ave)
    np.savetxt(fname + '_sem.txt', err_err)
    np.savetxt(fname + '_dt.txt', t)
    return
    
    
if __name__ == "__main__":
    main(sys.argv[1:])
