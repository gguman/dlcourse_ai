import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization, softmax


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
        self.fcl1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.relu = ReLULayer()
        self.fcl2 = FullyConnectedLayer(hidden_layer_size, n_output)
        
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
        self.params(self.fcl2)
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        
        #import ipdb; ipdb.set_trace()
        ffcl1 = self.fcl1.forward(X)
        frelu = self.relu.forward(ffcl1)
        ffcl2 = self.fcl2.forward(frelu)
        
        loss, d_preds = softmax_with_cross_entropy(ffcl2, y)
        
        
        bfcl2 = self.fcl2.backward(d_preds)
        brelu = self.relu.backward(bfcl2)
        bfcl1 = self.fcl1.backward(brelu)
        
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        #raise Exception("Not implemented!")
        
        loss_reg_w1 = self.reg * np.sum(np.power(self.fcl1.W.value, 2))
        loss_reg_w2 = self.reg * np.sum(np.power(self.fcl2.W.value, 2))
        
        loss_reg_b1 = self.reg * np.sum(np.power(self.fcl1.B.value, 2))
        loss_reg_b2 = self.reg * np.sum(np.power(self.fcl2.B.value, 2))
        
        loss += loss_reg_w1
        loss += loss_reg_w2
        
        loss += loss_reg_b1
        loss += loss_reg_b2
        
        self.fcl1.W.grad = self.fcl1.W.grad  + 2 * self.reg * self.fcl1.W.value
        self.fcl2.W.grad = self.fcl2.W.grad  + 2 * self.reg * self.fcl2.W.value
        
        self.fcl1.B.grad += 2 * self.reg * self.fcl1.B.value
        self.fcl2.B.grad += 2 * self.reg * self.fcl2.B.value
        
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
        
        ffcl1 = self.fcl1.forward(X)
        frelu = self.relu.forward(ffcl1)
        ffcl2 = self.fcl2.forward(frelu)
        
        pred = softmax(ffcl2)
        pred = np.argmax(pred, axis=1)
        
        #raise Exception("Not implemented!")
        
        return pred

    def params(self, layer=None):
        if not layer:
            result = {
                'W1': self.fcl1.W, 'B1': self.fcl1.B,
                'W2': self.fcl2.W, 'B2': self.fcl2.B
            }
            return result
        else:
            layer.W.grad = np.zeros_like(layer.W.grad)
            layer.B.grad = np.zeros_like(layer.B.grad)


        #    # TODO Implement aggregating all of the params

        #    #raise Exception("Not implemented!")

        #    return result
        