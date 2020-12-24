import numpy as np


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    # TODO: Copy from previous assignment
    raise Exception("Not implemented!")

    return loss, grad


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # TODO copy from the previous assignment
    raise Exception("Not implemented!")
    return loss, dprediction


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO copy from the previous assignment
        raise Exception("Not implemented!")

    def backward(self, d_out):
        # TODO copy from the previous assignment
        raise Exception("Not implemented!")
        return d_result

    def params(self):
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO copy from the previous assignment
        raise Exception("Not implemented!")

    def backward(self, d_out):
        # TODO copy from the previous assignment
        
        raise Exception("Not implemented!")        
        return d_input

    def params(self):
        return { 'W': self.W, 'B': self.B }

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding


    def forward(self, X):
        
        self.X = X
        
        if self.padding: self.X =\
            np.pad(X, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)))
        
        batch_size, height, width, channels = self.X.shape

        #out_height = 0
        #out_width = 0

        out_height = height - self.filter_size + 1
        out_width = width - self.filter_size + 1
        
        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        #self.X = X
        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        #import ipdb; ipdb.set_trace()
        _W = self.W.value.reshape(-1, self.out_channels)
        
        self.result = np.zeros((batch_size, out_height, out_width, self.out_channels))
        
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement forward pass for specific location
                _X = self.X[:, x:x+self.filter_size, y:y+self.filter_size, :].reshape(batch_size, -1)
                self.result[:, x, y, :] = np.dot(_X, _W) + self.B.value
                
        return self.result
                
        #raise Exception("Not implemented!")


    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output

        _W = self.W.value.reshape(-1, self.out_channels)
        d_input = np.zeros_like(self.X)
        
        # Try to avoid having any other loops here too
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                #pass
                #import ipdb; ipdb.set_trace()
                _X = self.X[:, x:x+self.filter_size, y:y+self.filter_size, :].reshape(batch_size, -1)
                
                dX = np.dot(d_out[:, x, y, :], _W.T)\
                    .reshape(batch_size, self.filter_size, self.filter_size, channels)
                dW = np.dot(_X.T, d_out[:, x, y, :])\
                    .reshape(self.filter_size, self.filter_size, self.in_channels, self.out_channels)
                dB = d_out[:, x, y, :].sum(axis=0)
                
                d_input[:, x:x+self.filter_size, y:y+self.filter_size, :] += dX
                self.W.grad += dW
                self.B.grad += dB
        
        if self.padding: d_input = d_input[:, self.padding:-self.padding, self.padding:-self.padding, :]
        
        return d_input
                
        #raise Exception("Not implemented!")

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        
        self.X = X
        
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        #raise Exception("Not implemented!")
        
        #out_height = int(self.pool_size / self.stride)
        #out_width = int(self.pool_size / self.stride)
        
        self.out_height = int((height - self.pool_size) / self.stride) + 1
        self.out_width = int((width - self.pool_size) / self.stride) + 1
        
        self.result = np.zeros((batch_size, self.out_width, self.out_height, channels))
        
        for x in range(self.out_width):
            for y in range(self.out_height):
                for b in range(batch_size):
                    for c in range(channels):
                        x_start = x * self.stride 
                        x_end = x * self.stride + self.pool_size
                        
                        y_start = y * self.stride
                        y_end = x * self.stride + self.pool_size
                        
                        self.result[b, x, y, c] =\
                            np.max(X[b, x_start:x_end, y_start:y_end, c])
        
        return self.result
        

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        #raise Exception("Not implemented!")
        
        d_input = np.zeros_like(self.X)
        
        for x in range(self.out_width):
            for y in range(self.out_height):
                for b in range(batch_size):
                    for c in range(channels):
                        x_start = x * self.stride 
                        x_end = x * self.stride + self.pool_size
                        
                        y_start = y * self.stride
                        y_end = x * self.stride + self.pool_size
                        
                        d_input[b, x_start:x_end, y_start:y_end, c] =\
                            np.isin(
                                self.X[b, x_start:x_end, y_start:y_end, c],
                                self.result[b, x, y, c]) * d_out[b, x, y, c]
        return d_input
        

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        raise Exception("Not implemented!")

    def backward(self, d_out):
        # TODO: Implement backward pass
        raise Exception("Not implemented!")

    def params(self):
        # No params!
        return {}
