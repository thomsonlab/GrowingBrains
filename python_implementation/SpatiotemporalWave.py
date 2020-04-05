## spatiotemporal wave generator in layer - I (square shaped layer)

# cdist in python is pdist2 in matlab, pdist in python is pdist in matlab
#
# remember * in matlab is matrix multiplication if its two matrices. in numpy have
# to use @ for matrix mult and * for element wise

# each point/entry in the matrix represents a node

# initialize retinal nodes and their properties
import numpy as np
from scipy.spatial.distance import squareform, pdist, cdist
import matplotlib.pyplot as plt


# had to install default colormap parula by doing pip install viscm
# then used this script: https://github.com/BIDS/colormap/blob/master/parula.py
# actual colormap is parula_map

from parula_script import parula_map


def matlab_percentile(x, p):
    p = np.asarray(p, dtype=float)
    n = len(x)
    p = (p-50)*n/(n-1) + 50
    p = np.clip(p, 0, 100)
    return np.percentile(x, p)

numRetina = 1
totNeurons_Retina = 1500
squareLen = 28
retinaParams_old = {'numNeurons':'', 'x':'', 'a':'', 'b':'', 'c':'', 'd':'', 'D':'', 'Dk':'', 'v':'', 'u':'',
                     'firings':'', 'array_act':'', 'fired':'', 'I':''}
retinaParams_old['numNeurons'] = totNeurons_Retina   # there are 1500 neurons

re = np.random.rand(totNeurons_Retina, 1)   #re is a 1500x1 array of random numbers from a random distribution over (0,1)


retinaParams_old['x'] = squareLen * np.random.rand(totNeurons_Retina, 2) # x is the 1500x2 array of random numbers from
                                                                        # (0,1) uniform distribution that is scaled by 28,
                                                                        # these are the values of the neurons at any given point (positions)
centroid_RGC = np.array([np.mean(retinaParams_old['x'], axis = 0)])  # centroid RGC is a 1x2 array that is the column mean of x


retinaParams_old['a'] = 0.02 * np.ones((totNeurons_Retina, 1), dtype=int) # a is .02 for every node
retinaParams_old['b'] = 0.2 * np.ones((totNeurons_Retina, 1), dtype = int) # b is .2 for every node
retinaParams_old['c'] = -65 + 15 * (re**2) # 1500x1 array

retinaParams_old['d'] = 8 - 6 * (re**2)
# retinaParams_old['d'] = np.full((3200,1),8) - gaussian_val
retinaParams_old['D'] = squareform(pdist(retinaParams_old['x']))
D = retinaParams_old['D']
retinaParams_old['Dk'] = np.multiply(5* (D<2).astype(int)- 2*(D>6).astype(int) , np.exp(-D/10))
retinaParams_old['Dk'] = retinaParams_old['Dk'] - np.diag(np.diag(retinaParams_old['Dk']))
retinaParams_old['v'] = -65 * np.ones((totNeurons_Retina, 1), dtype = int) # initial values of v, -65 for all nodes
retinaParams_old['u'] = retinaParams_old['b'] * retinaParams_old['v'] # recovery of sensory node i
retinaParams_old['firings'] = np.array([0, 0])   # will delete this 0,0 row afterwards
retinaParams_old['firings'].shape = (1,2)

fnum = 1
retinaParams = retinaParams_old

a = np.array([4])
outerRadius = a

# Controlling wave-size by altering sensor-node connectivity in layer-I!
retinaParams['Dk'] = (5* (D<2).astype(int)- 2*(D>outerRadius).astype(int) * np.exp(-D/10))
retinaParams['Dk'] = retinaParams['Dk'] - np.diag(np.diag(retinaParams_old['Dk']))

totTime = 3000 # Total time of simulation

pairWise_allRGC = np.sum(pdist(retinaParams['x']))
heatMap_wave = np.zeros((totNeurons_Retina,1)) # of times each neuron spikes

numActiveNodes_wave = np.array([])
clusterDisp_wave = np.array([])
radius_wave = np.array([])

for t in range(1, totTime + 1): # simulation of totTime (ms)
    print(t)
    spikeMat = np.zeros((totNeurons_Retina, 1)) # reset to zero after each period of totTime

    retinaParams['array_act'] = np.array([]) # active nodes reset to none
    retinaParams['I'] = np.array(3 * np.random.randn(totNeurons_Retina, 1)) # Noisy input, 1500x1 random numbers scaled by 3
    f = np.argwhere(retinaParams['v'] >= 30) # argwhere gives an arrray with the [row, column] values of
    g = []
    for i in range(len(f)):
        g.append(f[i][0]) # since retinaParams['v'] is 1500x1, this is a list of row indices of the ones that

    retinaParams['fired'] = np.array(g, dtype = np.int)
    retinaParams['fired'].shape = (len(g),1)

    fired = retinaParams['fired']

    retinaParams['array_act'] = retinaParams['x'][fired][:,0,:] # array_act are the ones that are fired
    retinaParams['fired'].shape = (len(retinaParams['fired']), 1)
    spikeMat[fired] = 1

    # Find number of nodes that fired AND cluster contiguity
    if len(retinaParams['array_act']) > 0:
        pairWise_firingNode = np.sum(pdist(retinaParams['array_act']))
    else:
        pairWise_firingNode = 0
    contig_firing = -np.log(pairWise_firingNode / pairWise_allRGC)

    if len(fired) > 10:
        heatMap_wave[fired] = heatMap_wave[fired]+1

        # plotting black position points
        plt.figure(1)
        plt.scatter(retinaParams['x'][:, 1], retinaParams['x'][:, 0], c='k')

        if np.shape(retinaParams['array_act'])[1] != 0:
            plt.scatter(retinaParams['array_act'][ :, 1],retinaParams['array_act'][ :, 0], color= 'red') # indexing is weird, but array act is # arrays x columns x rows. should fix probably
            plt.axis('off')
            plt.pause(.03)

    fnum += 1

    retinaParams['firings'] = np.append(retinaParams['firings'], np.concatenate([t + 0 * fired, fired], axis = 1), axis = 0)

    retinaParams['v'].shape = retinaParams['c'].shape

    retinaParams['d'].shape = 1500,1
    if len(fired) > 0:
        retinaParams['v'][fired] = retinaParams['c'][fired]
        retinaParams['u'][fired] = retinaParams['u'][fired] + retinaParams['d'][fired]

    # print(type(retinaParams['I'])
    retinaParams['I'].shape = (1500,1)
    retinaParams['v'].shape = (1500,1)
    retinaParams['I'] = retinaParams['I'] + np.sum(retinaParams['Dk'][:, fired], 1) # check what sum does
    retinaParams['v'] = retinaParams['v'] + 0.5 * (0.04 * retinaParams['v']** 2 + 5 * retinaParams['v'] + 140 - retinaParams['u'] + retinaParams['I'])

    retinaParams['v'] = retinaParams['v'] + 0.5 * (0.04 * retinaParams['v']** 2 + 5 * retinaParams['v'] + 140 - retinaParams['u'] + retinaParams['I'])

    retinaParams['u'] = retinaParams['u'] + retinaParams['a'] * (retinaParams['b']* retinaParams['v'] - retinaParams['u'])



    # keeping track of certain parameters:
    # (1): number of active nodes, (2): active nodes contiguity, (3): wave radius
    # to add (1), if nothing was fired, then we add 0 for that part of the loop
#    numActiveNodes_wave[t] = len(fired)

    numActiveNodes_wave = np.append(numActiveNodes_wave, [len(fired)])
    numActiveNodes_wave.shape = (len(numActiveNodes_wave), 1)

    clusterDisp_wave = np.append(clusterDisp_wave, contig_firing) # does having the ' make a difference
    clusterDisp_wave.shape = (len(clusterDisp_wave), 1)
    # need to have a better way of initially making the correct shape

    centroid_wave = np.mean(retinaParams['array_act'], 0) # check np.mean is okay for this
    centroid_wave.shape = (1, len(centroid_wave))

    dist = cdist(retinaParams['array_act'], centroid_wave)

    if len(dist) > 0:
        radius_wave = np.append(radius_wave, [centroid_wave[0][0], centroid_wave[0][1], matlab_percentile(dist, 80)])
    else:
        radius_wave = np.append(radius_wave, [centroid_wave[0][0], centroid_wave[0][1], np.nan])

    # note: np.percentile does not give the same percentile value as matlab's prctile,
    # this is because matlab uses midpoint interpolation and numpy and R use linear interpolation
    # so we make a function above to make matlab type percentiles (from stackoverflow)

print('went to end')

retinaParams['firings'] = np.delete(retinaParams['firings'] ,0, 0) # getting rid of dummy row


#### PLOTS ##########

# raster plot
#
plt.figure(2)

plt.scatter(retinaParams['firings'][:, 0], retinaParams['firings'][:, 1],
             c='b',
             marker = '.',
             edgecolors = 'none')
plt.xlabel('Time')
plt.ylabel('Neurons')
plt.title('Raster plot')

# HEATMAP WITH TIME (CUMULATIVE SPIKES OVER ENTIRE SIMULATION)

# heatMap_wave(heatMap_wave == 0) = 0.1;
#fig4.scatter(retinaParams['x'][:, 2], retinaParams['x'][:, 1], [], heatMap_wave[:, end], 'filled')
#colorbar
#title('probability of neuron firing - hotspots of wave')



#
plt.figure(3)
plt.scatter(retinaParams['x'][:, 1], retinaParams['x'][:, 0],
            c=heatMap_wave,
            cmap = parula_map,
            edgecolors = 'none')
plt.colorbar()
plt.title('probability of neuron firing - hotspots of wave')
plt.show()
