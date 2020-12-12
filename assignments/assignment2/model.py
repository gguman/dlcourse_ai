import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # TODO Create necessary layers
        self.fcl1 = FullyConnectedLayer(n_input, n_output)
        #raise Exception("Not implemented!")

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        #raise Exception("Not implemented!")
        
        self.params(self.fcl1)
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        
        ffcl1 = self.fcl1.forward(X)
        
        loss, d_preds = softmax_with_cross_entropy(ffcl1, y)
        
        bfcl1 = self.fcl1.backward(d_preds)
        

        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        #raise Exception("Not implemented!")
        
        loss_reg = self.reg * np.sum(np.power(self.fcl1.W.value, 2))
        grad_reg = 2 * self.reg * self.fcl1.W.value

        loss += loss_reg
        self.fcl1.W.grad += grad_reg
        
        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], np.int)

        raise Exception("Not implemented!")
        return pred

    def params(self, layer=None):
        if not layer:
            result = {'W': self.fcl1.W, 'B': self.fcl1.B}
            return result
        else:
            layer.W.grad = np.zeros_like(layer.W.grad)
            layer.B.grad = np.zeros_like(layer.B.grad)


        #    # TODO Implement aggregating all of the params

        #    #raise Exception("Not implemented!")

        #    return result
        