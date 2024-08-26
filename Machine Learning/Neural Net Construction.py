import random
from tqdm import tqdm
import numpy as np


class Network(object):
    def __init__(self, sizes):
        """
        Args:
            sizes (List[int]): Contains the size of each layer in the network.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    """
    4.1 Feed forward the input x through the network.
    """

    def feedforward(self, x):
        """
        Args:
            x (npt.array): Input to the network.
        Returns:
            List[npt.array]: List of weighted input values to each node
            List[npt.array]: List of activation output values of each node
        """
        #creating the lists to return
        a = x
        zs = []
        activations = [a]

        #computing the z and a values for each layer
        for l in range(1, self.num_layers):
            z = np.dot(self.weights[l-1], a) + self.biases[l-1]
            a = sigmoid(z)
            zs.append(z)
            activations.append(a)

        return zs, activations,

    """
    4.2 Backpropagation to compute gradients.
    """

    def backprop(self, x, y, zs, activations):
        """
        Args:
            x (npt.array): Input vector.
            y (float): Target value.
            zs (List[npt.array]): List of weighted input values to each node.
            activations (List[npt.array]): List of activation output values of each node.
        Returns:
            List[npt.array]: List of gradients of bias parameters.
            List[npt.array]: List of gradients of weight parameters.
        """
        #creating the lists to be returned
        deltaB = [np.zeros_like(b) for b in self.biases]
        deltaW = [np.zeros_like(w) for w in self.weights]

        #computing the gradient of the final layer
        aL = activations[-1]
        zL = zs[-1]
        deltaL = self.loss_derivative(aL, y) * sigmoid_prime(zL)
        deltaB[-1] = deltaL
        deltaW[-1] = np.outer(deltaL, activations[-2].transpose())

        #computing parameter gradients
        for l in range(2, self.num_layers):
            zL = zs[-l]
            deltaL = np.dot(self.weights[-l + 1].transpose(), deltaL) * sigmoid_prime(zL)
            deltaB[-l] = deltaL
            deltaW[-l] = np.outer(deltaL, activations[-l - 1].transpose())

        return (deltaB, deltaW)
    """
    4.3 Update the network's biases and weights after processing a single mini-batch.
    """

    def update_mini_batch(self, mini_batch, alpha):
        """
        Args:
            mini_batch (List[Tuple]): List of (input vector, output value) pairs.
            alpha: Learning rate.
        Returns:
            float: Average loss on the mini-batch.
        """
        #creating lists to store the updated weights
        cumulativeB = [np.zeros_like(b) for b in self.biases]
        cumulativeW = [np.zeros_like(w) for w in self.weights]
        cumulativeLoss = 0
    
        for x,y in mini_batch:
            #retrieving gradient values using feedforward and backprop
            zs, activations = self.feedforward(x)
            deltaB, deltaW = self.backprop(x, y, zs, activations)
            #computing cumulative gradients and loss
            for i in range(len(cumulativeB)):
                cumulativeB[i] += deltaB[i]
            for i in range(len(cumulativeW)):
                cumulativeW[i] += deltaW[i]                                        
            loss = self.loss_function(y, activations[-1])
            cumulativeLoss += loss

        #updating the weights
        n = len(mini_batch)
        for i in range(len(self.biases)):
            self.biases[i] -= (alpha/n) * cumulativeB[i]
        for i in range(len(self.weights)):
            self.weights[i] -= (alpha/n) * cumulativeW[i]

        #computing average loss 
        avgLoss = cumulativeLoss/n

        return avgLoss


    """
    Train the neural network using mini-batch stochastic gradient descent.
    """

    def SGD(self, data, epochs, alpha, decay, batch_size=32, test=None):
        n = len(data)
        losses = []
        for j in range(epochs):
            print(f"training epoch {j+1}/{epochs}")
            random.shuffle(data)
            for k in tqdm(range(n // batch_size)):
                mini_batch = data[k * batch_size : (k + 1) * batch_size]
                loss = self.update_mini_batch(mini_batch, alpha)
                losses.append(loss)
            alpha *= decay
            if test:
                print(f"Epoch {j+1}: eval accuracy: {self.evaluate(test)}")
            else:
                print(f"Epoch {j+1} complete")
        return losses

    """
    Returns classification accuracy of network on test_data.
    """

    def evaluate(self, test_data):
        test_results = [
            (np.argmax(self.feedforward(x)[1][-1]), y) for (x, y) in test_data
        ]
        return sum(int(x == y) for (x, y) in test_results) / len(test_data)

    def loss_function(self, y, y_prime):
        return 0.5 * np.sum((y - y_prime) ** 2)

    """
    Returns the gradient of the squared error loss function.
    """

    def loss_derivative(self, output_activations, y):
        return output_activations - y


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


    
