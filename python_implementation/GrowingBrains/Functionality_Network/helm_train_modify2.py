import scipy.stats as sts
import numpy as np
from mapminmax import mapminmax, mapminmax_apply
import time
import scipy
from result_tra import result_tra



def helm_train(train_x,train_y,test_x,test_y,b1,b2,b3,s,C):
    t = time.time()
    train_x = sts.zscore(train_x.T, ddof = 1).T # get zscore for each sample
    H1 = np.hstack((train_x, .1 * np.ones((train_x.shape[0],1)))) # hidden layer

    # first layer RELM

    A1 = np.dot(H1, b1)
    A1 = mapminmax(A1)
    beta1 = b1

    T1 = np.dot(H1, beta1)
    print('Layer 1: Max Val of Output {} Min Val {}'.format(np.max(T1), np.min(T1)))

    [T1, ps1] = mapminmax(T1.T, 0, 1)
    T1 = T1.T

    l1 = np.max(T1)
    l1 = s / l1
    print('Layer 3: Max Val of Output {} Min Val {}n'.format(l1, np.min(T1)))

    T1 = np.tanh(T1 * l1)

    beta = scipy.linalg.solve(np.dot(T1.T, T1) + (np.eye(T1.T.shape[0]) *(C)) , np.dot(T1.T , train_y))


    # beta here is different from in matlab, even though breaking it up gives very very similar values
    # for A and B in Ax = B, which is what we are solving for here (solving for beta as x)
    part1 = (np.dot(T1.T, T1) + (np.eye(T1.T.shape[0]) *(C)))
    part2 = np.dot(T1.T , train_y)
    part3 = np.linalg.solve(part1,part2)
    part4 = np.dot(np.linalg.inv(part1), part2)

    Training_time = time.time() - t
    print('Training has been finished!')
    print('The Total Training Time is : {} seconds'.format(str(Training_time)))

    # Calculate the training accuracy
    xx = np.dot(T1, beta)
    yy = result_tra(xx)                     # output (classification) based on our network
    train_yy = result_tra(train_y)          # actual output of the sample
    TrainingAccuracy = len(np.argwhere(yy == train_yy)) / float(train_yy.shape[0])
    print('Training Accuracy is : {}'.format(str(TrainingAccuracy * 100)) + ' %')


    # First layer Feedforward

    t = time.time()
    test_x = sts.zscore(test_x.T, ddof = 1).T # get zscore for each sample
    HH1 = np.hstack((test_x, .1 * np.ones((test_x.shape[0],1)))) # hidden layer
    TT1 = np.dot(HH1 , beta1)
    TT1 = mapminmax_apply(TT1.T, ps1.xrange, ps1.xmin, ps1.yrange,ps1.ymin).T
    TT1 = np.tanh(TT1 * l1)
    x = np.dot(TT1, beta)
    y = result_tra(x)
    test_yy = result_tra(test_y)
    TestingAccuracy = len(np.argwhere(y == test_yy)) / float(test_yy.shape[0])


    # Calculating the testing accuracy
    Testing_time = time.time() - t
    print('Testing has been finished!')
    print('The Total Testing Time is : ' + str(Testing_time) + ' seconds')
    print('Testing Accuracy is : ' + str(TestingAccuracy * 100) + ' %');
    return TrainingAccuracy, TestingAccuracy, Training_time, Testing_time