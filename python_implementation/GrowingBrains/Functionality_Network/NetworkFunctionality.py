import scipy.io as sio
import numpy as np
from helm_train_modify2 import helm_train

# this code is to couple the two layered network with a linear classifier to classify hand-written digits from MNIST
# on the basis of the representation provided by hand-crafted, self-organized, and random networks
# RELM is a linear classifier

numSim = 1
train_testAcc = np.zeros((numSim,6))
for numSim in range(1, numSim+1):
    # loading training and testing data from matlab files
    train_matlab = sio.loadmat("../../../Functionality_Network/train_testMNIST/MNIST_train_{}.mat".format(numSim))
    test_matlab = sio.loadmat("../../../Functionality_Network/train_testMNIST/MNIST_test_{}.mat".format(numSim))

    train_x = train_matlab['train_MNIST']
    train_y = train_matlab['labels_train']

    test_x = test_matlab['test_MNIST']
    test_y = test_matlab['labels_test']

    # loading weights of networks from basisVec # todo: where does basis vec come from originally?
    basisVec = sio.loadmat("../../../Functionality_Network/basisVec/basisVec_MNIST_{}.mat".format(numSim))
    s2Matrix = basisVec['s2Matrix']
    synapticMatrix = basisVec['synapticMatrix']
    randomSynMatrix = basisVec['randomSynMatrix']


    # RANDOMNESS

    usefulInd = np.argwhere(np.sum(s2Matrix, axis=0) < 150).T  # find columns whose sums are < 150 in the weight matrix
    N1 = len(usefulInd[0])
    N2 = 10

    # HARD CODED POOLING

    pos = np.random.choice(synapticMatrix.shape[1], N1, replace=False) # from the row vector 1:441, pick x numbers randomly sampled without replacement
    b1_hc = synapticMatrix[:,pos]
    b1_hc = np.vstack((b1_hc, np.random.rand(1,N1)))         # add a row beneath b1_hc of random numbers btwn 0 and 1



    # SELF ORGANIZED POOLING

    b1_selfOrg = s2Matrix[:,usefulInd[0]]
    b1_selfOrg = np.vstack((b1_selfOrg, np.random.rand(1, N1)))


    # RANDOMIZED

    pos = np.random.choice(randomSynMatrix.shape[1],N1,replace=False)
    b1_rand = randomSynMatrix[:,pos]
    b1_rand = np.vstack((b1_rand, np.random.rand(1,N1)))

    b2 = 2 * np.random.rand(N1+1,N2)-1
    b3 = []


    C = 2^-30
    s = 1

    #Call the training function

    [Training_hc, Testing_hc, Training_time, Testing_time] = helm_train((train_x), (train_y), (test_x), (test_y), b1_hc, b2, b3, s, C)
    print('')
    [Training_selfOrg, Testing_selfOrg, Training_time, Testing_time] = helm_train((train_x), (train_y), (test_x), (test_y), b1_selfOrg, b2, b3, s, C)
    print('')
    [Training_rand, Testing_rand, Training_time, Testing_time] = helm_train((train_x), (train_y), (test_x), (test_y), b1_rand, b2, b3, s, C)

    train_testAcc = np.vstack((train_testAcc, [Training_hc*100, Testing_hc*100,Training_selfOrg*100, Testing_selfOrg*100,Training_rand*100, Testing_rand*100]))



