import numpy as np

def result_tra(x):
    ''' takes in output neurons of several samples and gives the index
     of the neuron with the highest value for each sample corresponding to its MNIST number'''

    y = np.zeros(((x.shape)[0],1))
    for i in range(x.shape[0]):
        y[i] = np.argmax(x[i]) + 1

    return y


# for x = 10,000 x 10
#     y = 10,000 x 1
# for each row, which is one sample, gets whichever index is the highest in the row, that corresponds to then nuner