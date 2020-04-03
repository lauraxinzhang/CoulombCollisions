import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib
import matplotlib.pylab as pylab

def updateGlobal(fontsize):
    params = {'legend.fontsize': fontsize-2,
          'figure.figsize': (6, 4),
         'axes.labelsize': fontsize,
         'axes.titlesize':fontsize+2,
         'xtick.labelsize':fontsize,
         'ytick.labelsize':fontsize,
         'font.size': fontsize}
    pylab.rcParams.update(params)

def formatAndSave(fig, ax, lines, xlabel, ylabel, legends = None, title = None, fontsize = 14, filename = None):
    '''
    formats the current figure, and save to file if filename is provided.
    '''
    
    fig.set_size_inches(6, 4)
    plt.tight_layout()
    
    updateGlobal(fontsize)

    if legends == None:
#       use labels provides by line labels
        leg = ax.legend()
    elif legends == False:
        leg = None
    else:
        leg = ax.legend(legends)
    
    if leg != None:
        for lh in leg.legendHandles: 
            lh.set_alpha(1)
        
    ax.set_xlabel(xlabel, fontsize = fontsize)
    ax.set_ylabel(ylabel, fontsize = fontsize)
    plt.setp(lines, linewidth = 2)
#     plt.ticklabel_format(axis = 'y', style = 'sci', scilimits=(0, 0))
    plt.setp(ax.get_xticklabels(), fontsize=fontsize-2)
    plt.setp(ax.get_yticklabels(), fontsize=fontsize-2)
    
#     ax.setp(ax.spines.values(), linewidth=5)
    if title != None:
        ax.set_title(title, fontsize = fontsize + 3)
    if filename != None:
        fig.savefig(filename, bbox_inches = "tight")
    
    
def aveAndSTD(ax, vList, tTot, markers = '-', label = None):
    '''
    Plots the mean +- std of vList
    ax: axes to plot on
    vList: np array containing values to be plotted; average is taken along the -1 axis
    tTot: total simulation time
    markers: default '-'
    label: to be used by legends
    return: lines objects returned by plt.plot
    '''
    ave = np.mean(vList, axis = -1)
    std = np.std(vList, axis = -1)
    
    avediff = (ave - ave[0]) / ave[0]
    aveplus = avediff + std / ave[0]
    aveminus = avediff - std / ave[0]
    
    time = np.linspace(0, tTot, vList.shape[0])
    
    lines = ax.plot(time, avediff, markers, label = label)
    ax.fill_between(time, aveminus, aveplus, alpha = 0.2)
    return lines

def vSpaceSnaps(ax, vHist, tslices, tTot):
    '''
    Plots where the particles are in velocity space
    ax: axes to plot on
    vHist: evolution of v vectors in time
    tslices [s]: a list of time slices to plot on
    tTot: total simulation time
    '''
    plt.ticklabel_format(axis = 'x', style = 'sci', scilimits=(-2, 2), useMathText=True)
    plt.ticklabel_format(axis = 'y', style = 'sci', scilimits=(-2, 0), useMathText=True)
    
    length = vHist.shape[0]
    dt = tTot/length
    indices = np.floor(tslices / dt)
    print(indices)
#     lines = []
    plt.axis('equal')
    
    colors = pylab.cm.plasma(np.linspace(0,1,len(indices)))
    for num in range(len(indices)):
#     for i in reversed(indices):
        i = indices[len(indices)-1-num]
        if i >= length:
            i = length-1
        i = int(i)
        vx = vHist[i, :, 0]
        vy = vHist[i, :, 1]
        vz = vHist[i, :, 2]
        vperp = np.sqrt(vy**2 + vz**2) * np.sign(vz)
        lines = ax.scatter(vx, vperp, alpha = 0.2, 
                           label = str(tslices[len(indices)-1-num]) + ' s', 
                           color = colors[len(indices)-1-num])
    return lines
    
    
def plotConvergence(ax, x_data, y_data):
    if len(y_data.shape) ==1:
        lines = ax.scatter(x_data, y_data)
    else:
        for i in range(y_data.shape[-1]):
            lines = ax.scatter(x_data, y_data[:, i])
        
    return lines