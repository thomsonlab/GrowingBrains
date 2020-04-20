#
import numpy as np
from scipy.spatial.distance import squareform, pdist, cdist, euclidean
import matplotlib.pyplot as plt

# test[test>2] = test[test>2] * 2 using actual values of test and changing them based on index
###############################################################################
# initialize retinal nodes and their properties:
###############################################################################

numSim = 1  # does this have to be a loop or not? bc u clear variables afterwards
numRetina = 1
totNeurons_Retina = 3200
squareLength = 32

class Neuron: # to make each neuron, replaces retinaParams_old

    # Class Attribute

    # Initializer / Instance Attributes
    def __init__(self, name, x = np.zeros((1,2)), d = 0, D = np.array([]), Dk = np.array([]), a = 0.02, b = 0.2, c = -65 + (15 * np.random.rand()**2),
                  v = -65, u = -13, firings = [],dist_center_to_all = 0, gaussian_val = 0, x_3d = np.array([])):
        self.ID = name # number of neuron.
        self.x = x # (x, y) value for position. 32 * random number between 0 and 1 aka position
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.D = D # 1 x numNeurons array with the distance between self and neuron in column x as value
        self.Dk = Dk
        self.v = v # default is -65
        self.u = u # self.v * self.b
        self.firings = firings
        self.dist_center_to_all = dist_center_to_all
        self.gaussian_val = gaussian_val
        self.x_3d = x_3d
        self.synapticMatrix_retinaLGN = np.array([])# weights between first layer and second layer, maybe shouldnt be an attribute of this class?
        self.s2Matrix = np.array([])

# idea could also be to have the




# to make a neuron, give all the fields
# for given numNeurons, make all the neurons with the right ID
# make a D by making a 1x numNeuron array putting distance between self.x and the position of each other neuron
# making all neurons. call a neurons by doing retinaNeurons[1] --> totNeurons_Retina[numNeurons -1]

retinaNeurons = {i: Neuron(name=i, x = squareLength * np.random.rand(1, 2)) for i in range(totNeurons_Retina)} # makes a dictionary of the neurons with each neuron having name with its number!
                                            # (x, y) value for position. 32 * random number between 0 and 1 aka position


# 1 x 2 mean of positions, go through each neuron and add the x value and get mean
x_sum = 0
y_sum = 0
for i in range(totNeurons_Retina):
    x_sum += retinaNeurons[i].x[0][0]
    y_sum += retinaNeurons[i].x[0][1]
centroid_RGC = np.array([[x_sum/totNeurons_Retina, y_sum/totNeurons_Retina]])

# make dist_center_to_all --> each neuron gets its field populated (distance from centroid to each neurons position)
for i in range(totNeurons_Retina):
    retinaNeurons[i].dist_center_to_all = cdist(retinaNeurons[i].x, centroid_RGC)
    retinaNeurons[i].gaussian_val = 6 * np.exp(-retinaNeurons[i].dist_center_to_all / 10)
    retinaNeurons[i].d = 8 - retinaNeurons[i].gaussian_val

# now to make D, which is the distance matrix between neurons. make a [n, ] matrix for each neuron gives its distance to all the others

# if we know what D is supposed to be in terms of size, we should make it with the right size so its not slow to append

for i in range(totNeurons_Retina):

    for j in range(totNeurons_Retina):
        retinaNeurons[i].D = np.zeros(totNeurons_Retina,)
        retinaNeurons[i].D[j] = euclidean(retinaNeurons[i].x, retinaNeurons[j].x)

    # once you make D, now you can make the weights, since they depend on D being completed (entire row filled)
    # making the weights in the right spots (5 or -2), then multiplying by e^original distance/10
    retinaNeurons[i].Dk = np.multiply(
        5 * (retinaNeurons[i].D < 2).astype(int) - 2 * (retinaNeurons[i].D > 5).astype(int),
        np.exp(-retinaNeurons[i].D / 10))

    # is this necessary isnt it all zeros anywway
    # retinaParams_old['Dk'] = retinaParams_old['Dk'] - np.diag(np.diag(retinaParams_old['Dk']))

    retinaNeurons[i].u = retinaNeurons[i].b * retinaNeurons[i].v # redundant

## same as:
# for neuron1 in retinaNeurons.values():
#
#     for neuron2 in retinaNeurons.values():
#         neuron1.D = np.append(neuron1.D, euclidean(neuron1.x, neuron2.x))
#





plt.figure(1)
for neuron in retinaNeurons.values():
    plt.scatter(neuron.x[0][0], neuron.x[0][1], c='k')
plt.show()


outerRadius = 6





## Change wave-size!

for i in range(totNeurons_Retina):

    retinaNeurons[i].Dk = np.multiply(
        5 * (retinaNeurons[i].D < 2).astype(int) - 2 * (retinaNeurons[i].D > outerRadius).astype(int),
        np.exp(-retinaNeurons[i].D / 10))
    # retinaParams_old['Dk'] = retinaParams_old['Dk'] - np.diag(np.diag(retinaParams_old['Dk']))

    retinaNeurons[i].x_3d = np.concatenate(np.concatenate((retinaNeurons[i].x, np.zeros([1,1])), axis=1)) # adds third dimension point as 0


## Parameters of the LGN

eta = 0.1
decay_rt = 0.01
maxInnervations = totNeurons_Retina
# maxInnervations = 1499

LGN_params = {}
connectedNeurons = []
initConnections  = []

mu_wts = 2.5
sigma_wts = 0.14


###############################################################################
# Choose random nodes on the arbitGeometry Retina -- and layer up!
###############################################################################
# randi(10,3,4) returns a 3-by-4 array of pseudorandom integers between 1 and 10.
# X = randi([imin,imax],___) returns an array containing integers drawn from the discrete uniform distribution on the
# interval [imin,imax], using any of the above syntaxes.

# make LGN layer
LGN_num = 400
numLGN = LGN_num

# makes a dictionary of the neurons with each neuron having name with its number!
# position is random position from retinaNeurons


layer_LGN = {i: Neuron(name=i, x = retinaNeurons[np.random.randint(low=1, high=totNeurons_Retina)].x) for i in range(LGN_num)}

for i in range(LGN_num):
    layer_LGN[i].x_3d = np.concatenate(
        np.concatenate((layer_LGN[i].x, np.ones([1, 1])), axis=1))  # adds third dimension point as 0

# make matrix of distance between 3d retina positions and 3d lgn positions
# 3200 x 400, i,j is the distance between retinaNeuron i and lgn neuron j,
# have to do it in a for loop because positions arent matrices

di = np.zeros((totNeurons_Retina, LGN_num)) # distance between retina Neurons and LGN neurons in 3d

for i in range(totNeurons_Retina):

    for j in range(numLGN):
        di[i][j] = euclidean(retinaNeurons[i].x_3d, layer_LGN[j].x_3d)



###############################################################################
# normalizing synaptic matrix
###############################################################################
synapticMatrix_retinaLGN = np.zeros((totNeurons_Retina, numLGN), dtype = int) # 3200 x 400 matrix of zeros, weights between retinaNeurons and lgn neurons


# synaptic matrix, it can be a matrix, because no redundant numbers, or it could be a 1 x 400 array
# for each retina neuron showing the weights between each retina neuron and each lgn in the second layer
# lets do it as a (1, 400) array for each neuron

for i in range(totNeurons_Retina):

    retinaNeurons[i].synapticMatrix_retinaLGN = np.random.normal(loc=mu_wts, scale=sigma_wts, size=(1, 400)) # weight between this retina neuron and LGN neuron
    retinaNeurons[i].synapticMatrix_retinaLGN = retinaNeurons[i].synapticMatrix_retinaLGN / np.mean(retinaNeurons[i].synapticMatrix_retinaLGN) * mu_wts


LGN_synapticChanges = np.zeros((numLGN, 1), dtype = int)
LGN_threshold = np.random.normal(loc = 70, scale = 2, size =(numLGN, 1))


LGNactivity = np.array([])
initSynapticMatrix_retinaLGN = synapticMatrix_retinaLGN

heatMap_wave = np.zeros((totNeurons_Retina, 1), dtype = int)   # of times each neuron spikes
rfSizes = np.c_[150:800:50]


###############################################################################
# Spontaneous synchronous bursts of Retina(s)
###############################################################################
t = 0

# tic # starts stopwatch timer

rgc_connected = []
time_rf = np.zeros((rfSizes.shape[0],1), dtype=int)

numRfcompInit = 0

#### modular part breaks down here: how to get sum of columns if we have the synaptic matrix as a 1x 400 thing for each layer I neuron?
#### instead could make each thing 1 x 400 instead and represent the s2 matrix as an attribute of the LGN neurons instead, so for
#### each LGN neuron, can know how many layer I neurons its connected to

## everything after this point assumes matrices

while True:
    t += 1
    if t > 1e6:
        break
    if t % 1000 == 0:
        for neuron in retinaNeurons.values():
            neuron.s2Matrix = neuron.synapticMatrix_retinaLGN # weights for each neuron

            neuron.s2Matrix[neuron.s2Matrix < 0.1] = np.NaN   # make values that are less than 0.1 be NaN

            neuron.s2Matrix = ~np.isnan(neuron.s2Matrix)   # set the array to be 0, 1 boolean of the positions that
                                            # dont equal Nan, like if it is NaN, puts 0, ifi ts nan, put 1 (true)

                                                   # dont equal Nan, like if it is NaN, puts 0, ifi ts not, put 1 (true)
                                                    # 0 if s2 < .1 (if connection is < .1


            for ind_rf in range(len(rfSizes)):
                rf = rfSizes[ind_rf]     # for every 150, 200, 250, etc.
                temp1 = np.where(sum(s2Matrix) < rf)   # finds index of columns where the sum of the column is < rf
                                                    # which means that that column (that lgn) is connected to less than 150 ( or 200, 300,
                                                    # etc) number of neurons in layer I
                # if more than numLGN number of columns are less than rf and timerf ==0
                if len(temp1) > 0.9*numLGN and time_rf[ind_rf] == 0:
                    time_rf[ind_rf] = t

            numRfcomp = len(time_rf)-len(np.where(time_rf == 0))

            if numRfcomp >= 1:
                if numRfcomp > numRfcompInit:
                    u = np.where(time_rf != 0)[0]   # np.argwhere gives [row, column] values where this is the case. for example,
                                                    # could be [ [3,0], [5,0] ] where time rf is not 0 at 3rd row 0th
                                                    # column and 5th row zeroth column
                                                    # np.where gives [row, row], [column, column] tuples that are better for indexing.
                                                    # so this is giving us the row values that work

                    minRFsize = u[0]                # first row index that satisfies thing

                    s2Matrix = synapticMatrix_retinaLGN
                    s2Matrix = s2Matrix.astype('float')
                    s2Matrix[s2Matrix < 0.1] = np.NaN
                    s2Matrix = ~np.isnan(s2Matrix)

                    temp1 = np.where(sum(s2Matrix) < rfSizes[minRFsize]) # which columns have a sum less than 150
                    temp1 = temp1[0]

                    plt.figure(2)

                    ctr = 1

