import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization, softmax_with_cross_entropy, softmax
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        # TODO Create necessary layers
        #raise Exception("Not implemented!")
        
        image_width, image_height, n_channels = input_shape
        
        self.conv1 = ConvolutionalLayer(n_channels, conv1_channels, 3, 1)
        self.relu1 = ReLULayer()
        self.maxp1 = MaxPoolingLayer(4, 4)
        self.conv2 = ConvolutionalLayer(conv1_channels, conv2_channels, 3, 1)
        self.relu2 = ReLULayer()
        self.maxp2 = MaxPoolingLayer(4, 4)
        self.flatn = Flattener()
        
        fc_input = int(image_width * image_height * conv2_channels / pow(4, 4))
        self.fc = FullyConnectedLayer(fc_input, n_output_classes)
        

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        
        self.conv1.W.grad = np.zeros_like(self.conv1.W.grad)
        self.conv1.B.grad = np.zeros_like(self.conv1.B.grad)
        
        self.conv2.W.grad = np.zeros_like(self.conv2.W.grad)
        self.conv2.B.grad = np.zeros_like(self.conv2.B.grad)
        
        self.fc.W.grad = np.zeros_like(self.fc.W.grad)
        self.fc.B.grad = np.zeros_like(self.fc.B.grad)
        
        # TODO Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment
        #raise Exception("Not implemented!")
        
        fconv1 = self.conv1.forward(X)
        frelu1 = self.relu1.forward(fconv1)
        fmaxp1 = self.maxp1.forward(frelu1)
        
        fconv2 = self.conv2.forward(fmaxp1)
        frelu2 = self.relu2.forward(fconv2)
        fmaxp2 = self.maxp2.forward(frelu2)
        
        fflatn = self.flatn.forward(fmaxp2)
        
        ffc = self.fc.forward(fflatn)
        
        loss, d_preds = softmax_with_cross_entropy(ffc, y)
        
        bfc = self.fc.backward(d_preds)
        
        bflatn = self.flatn.backward(bfc)
        
        bmaxp2 = self.maxp2.backward(bflatn)
        brelu2 = self.relu2.backward(bmaxp2)
        bconv2 = self.conv2.backward(brelu2)
        
        bmaxp1 = self.maxp1.backward(bconv2)
        brelu1 = self.relu1.backward(bmaxp1)
        bconv1 = self.conv1.backward(brelu1)
        
        return loss
        
        
        
        
        

    def predict(self, X):
        # You can probably copy the code from previous assignment
        #raise Exception("Not implemented!")
        fconv1 = self.conv1.forward(X)
        frelu1 = self.relu1.forward(fconv1)
        fmaxp1 = self.maxp1.forward(frelu1)
        
        fconv2 = self.conv2.forward(fmaxp1)
        frelu2 = self.relu2.forward(fconv2)
        fmaxp2 = self.maxp2.forward(frelu2)
        
        fflatn = self.flatn.forward(fmaxp2)
        
        ffc = self.fc.forward(fflatn)
        
        prob = softmax(ffc)
        
        pred = np.argmax(prob, axis=1)
        
        return pred
        
        

    def params(self):
        result = {
            'Wc1':self.conv1.W, 'Bc1': self.conv1.B,
            'Wc2':self.conv2.W, 'Bc2': self.conv2.B,
            'Wfc':self.fc.W, 'Bfc':self.fc.B
        }

        # TODO: Aggregate all the params from all the layers
        # which have parameters
        #raise Exception("Not implemented!")

        return result
