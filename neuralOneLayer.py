'''
https://iamtrask.github.io/2015/07/12/basic-python-network/
good at http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/
'''
import numpy as np
import matplotlib as mt
import matplotlib.pyplot as plt

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
    
# input dataset
X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])
    
# output dataset            
y = np.array([[0,0,1,1]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2*np.random.random((3,1)) - 1

for iter in xrange(100000):

    # forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))
    ##test_x = np.array([i for s in l0 for i in s])
    ##test_y = np.array([i for s in l1 for i in s])
    test_x = np.array(np.dot(l0,syn0))
    test_y = np.array(l1)    
    #print test_x
    #print '\n'*1
    #print test_y
    plt.plot(test_y,test_x)
    # how much did we miss?
    l1_error = y - l1

    # multiply how much we missed by the 
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1,True)

    # update weights
    syn0 += np.dot(l0.T,l1_delta)

print "Output After Training:"
print l1
plt.show()
