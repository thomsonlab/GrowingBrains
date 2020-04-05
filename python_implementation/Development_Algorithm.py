import numpy as np
from scipy.spatial.distance import cdist, squareform, pdist
import matplotlib.pyplot as plt
import math




## Implementing Developmental algorithm
#  1. Spatiotemporal wave generator in layer-I
#  2. Learning rule implemented in layer-II
#  ==> Results in a pooling architecture

# clear, clc, close all = clears variables,
# clears text from command window, and closes
# all open figures
# for python, we can just fake it and print new lines
# probably can add another thing for clearing variables

# __saved_context__ = {}
#
# def clear():
#     for name in dir():
#         if not name.startswith('_'):
#             del globals()[name]
#     return None
#
#
# def clc():
#     print('\n' * 80)
#
#
# def save(to_save):
#     __saved_context__.update({to_save: to_save})
#
# clear()
# clc()

numSim = 1  # does this have to be a loop or not? bc u clear variables afterwards

###############################################################################
# initialize retinal nodes and their properties:
###############################################################################


numRetina = 1
totNeurons_Retina = 3200
squareLength = 32
retinaParams_old = [{'numNeurons':'', 'x':'', 'a':'', 'b':'', 'c':'', 'd':'', 'D':'', 'Dk':'', 'v':'', 'u':'',
                     'firings':''}]


retinaParams_old['numNeurons'] = totNeurons_Retina
re = np.random.rand(totNeurons_Retina, 1)
retinaParams_old['x'] = squareLength * np.random.rand(totNeurons_Retina, 2)
centroid_RGC = np.array([np.mean(retinaParams_old['x'], axis = 0)])

# subtract the column mean (centroid_RGC) from retinaParams_old(i).x
dist_center_to_all = retinaParams_old['x'] - centroid_RGC

# whats the point of calculating the first dist_center_to_all if you're going to redo it anyway
# computed euclidean distance between the points of the rows of retinaParams_old.x and the column means,
# output is a 3200 x 1 array

dist_center_to_all = cdist(retinaParams_old['x'], centroid_RGC)
gaussian_val = 6 * np.exp(-(dist_center_to_all) / 10)
retinaParams_old['a'] = 0.02 * np.ones((totNeurons_Retina, 1), dtype=int)
retinaParams_old['b'] = 0.2 * np.ones((totNeurons_Retina, 1), dtype = int)
retinaParams_old['c'] = -65 + 15 * np.power(re, 2)

# retinaParams_old[0]['d'] = 8 - 6 * np.power(re, 2)
retinaParams_old['d'] = np.full((3200,1),8) - gaussian_val
retinaParams_old['D'] = squareform(pdist(retinaParams_old['x']))
D = retinaParams_old['D']
retinaParams_old['Dk'] = np.multiply(5* (D<2).astype(int)- 2*(D>5).astype(int) , np.exp(-D/10))
retinaParams_old['Dk'] = retinaParams_old[0]['Dk'] - np.diag(np.diag(retinaParams_old['Dk']))
retinaParams_old['v'] = -65 * np.ones((totNeurons_Retina, 1), dtype = int) # initial values of v
retinaParams_old['u'] = np.multiply(retinaParams_old['b'] , retinaParams_old['v'])
retinaParams_old['firings'] = np.array([])

# end

retinaParams = retinaParams_old

# scatterplot of  second column of x, first column of x, k (black),  filled (fills in the circles)

plt.figure(1)
plt.scatter(retinaParams_old['x'][:,1], retinaParams_old['x'][:,0], c = 'k')

#
# # pause(0.2)
#
# LGN_num = np.array([400])
#
# #for
# outerRadius = 6
# numLGN = LGN_num
#
# # change wave-size!
#
# retinaParams[0]['Dk'] = np.multiply(5*(D<2).astype(int)- 2*(D>outerRadius) , np.exp(-D/10).astype(int))
# retinaParams[0]['Dk'] = retinaParams[0]['Dk'] - np.diag(np.diag(retinaParams[0]['Dk']))
#
# x_3d = np.concatenate([retinaParams[0]['x'], np.zeros((len(retinaParams[0]['x']),1), dtype = int)], axis = 1)
#
# ###############################################################################
# # parameters of the LGN
# ###############################################################################
#
# eta = 0.1
# decay_rt = 0.01
# maxInnervations = totNeurons_Retina
# # maxInnervations = 1499
#
# LGN_params = {}
# connectedNeurons = []
# initConnections  = []
#
# mu_wts = 2.5
# sigma_wts = 0.14
#
# synapticMatrix_retinaLGN = np.zeros((totNeurons_Retina, numLGN), dtype = int)
#
#
#
# ###############################################################################
# # Choose random nodes on the arbitGeometry REtina -- and layer up!
# ###############################################################################
# # randi(10,3,4) returns a 3-by-4 array of pseudorandom integers between 1 and 10.
# # X = randi([imin,imax],___) returns an array containing integers drawn from the discrete uniform distribution on the
# # interval [imin,imax], using any of the above syntaxes.
#
# layer_LGN = np.random.randint(low=1, high=totNeurons_Retina, size = (numLGN, 1))
#
# LGN_pos2d = retinaParams[0]['x'][layer_LGN,:]
# LGN_pos2d.resize(400,2)
# LGN_pos3d = np.concatenate([LGN_pos2d, np.ones((len(LGN_pos2d),1), dtype = int)], axis =1)
#
# di = cdist(x_3d, LGN_pos3d)
#
# ###############################################################################
# # normalizing synaptic matrix
# ###############################################################################
# for i in range(400):
#     synapticMatrix_retinaLGN[:,i] = np.random.normal(loc = mu_wts, scale = sigma_wts,
#                                                      size = (totNeurons_Retina,))
#     synapticMatrix_retinaLGN[:,i] = synapticMatrix_retinaLGN[:,i]/np.mean(synapticMatrix_retinaLGN[:,i]) * mu_wts
#
# LGN_synapticChanges = np.zeros((numLGN, 1), dtype = int)
# LGN_threshold = np.random.normal(loc = 70, scale = 2, size =(numLGN, 1))
#
#
# LGNactivity = np.array([])
# initSynapticMatrix_retinaLGN = synapticMatrix_retinaLGN
#
# heatMap_wave = np.zeros((totNeurons_Retina, 1), dtype = int)   # of times each neuron spikes
# rfSizes = np.c_[150:750:50]
#
# ###############################################################################
# # Spontaneous synchronous bursts of Retina(s)
# ###############################################################################
# t = 0
#
# tic
# rgc_connected = []
# time_rf = zeros(size(rfSizes,1),1)
# numRfcompInit = 0
#
# while True:
#         t = t+1
#         if t> 1e6:
#             break
#         if np.mod(t, 1000) == 0:
#
#             s2Matrix = synapticMatrix_retinaLGN
#             s2Matrix(s2Matrix < 0.1) = np.NaN
#             s2Matrix = ~isnan(s2Matrix)