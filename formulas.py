import numpy as np

def sigmoid(z): 
    '''
    Input:
        z: is the input (can be a scalar or an array)
    Output:
        h: the sigmoid of z
    '''
    
    # calculate the sigmoid of z
    h = 1/(1 + np.exp(-z))
    
    return h

def gradientDescent(x, y, theta, alpha, num_iters):
    '''
    Input:
        x: matrix of features which is (m,n+1)
        y: corresponding labels of the input matrix x, dimensions (m,1)
        theta: weight vector of dimension (n+1,1)
        alpha: learning rate
        num_iters: number of iterations you want to train your model for
    Output:
        J: the final cost
        theta: your final weight vector
    '''
    # get 'm', the number of rows in matrix x
    m = len(x)
    
    for i in range(0, num_iters):
        
        # get z, the dot product of x and theta
        z = np.dot(x, theta)
        
        # get the sigmoid of z
        h = sigmoid(z)
        
        # calculate the cost function
        J = (-1./m) * (np.dot((np.transpose(y)), np.log(h)) + np.dot(((np.transpose(1-y))), np.log(1.-h)))

        # update the weights theta
        theta = theta - alpha * (1./m) * (np.dot(np.transpose(x), (h-y)))
        
    J = float(J)
    return J, theta
