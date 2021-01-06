import torch
from torchvision import datasets
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import optim
from mpl_toolkits.mplot3d import Axes3D
import time
from mpl_toolkits import mplot3d
import pandas as pd
import sys
import csv
import tensorflow as tf
from emnist import extract_training_samples
import csv

# Create a class for the actual neural network
class NeuralNetwork(nn.Module):

    def __init__(self):

        super().__init__()
        inputNeurons, hiddenNeurons, outputNeurons = 784, 800, 36

        # Create tensors for inputs and outputs
        #input = nn.Linear((1, inputNeurons))
        #output = nn.Linear((1, outputNeurons))

        # Create tensors for the weights
        self.layerOne = nn.Linear(inputNeurons, hiddenNeurons)
        self.layerTwo = nn.Linear(hiddenNeurons, hiddenNeurons)
        self.layerThree = nn.Linear(hiddenNeurons, hiddenNeurons)
        self.layerFour = nn.Linear(hiddenNeurons, hiddenNeurons)
        self.layerFive = nn.Linear(hiddenNeurons, outputNeurons)

    # Create function for Forward propagation
    def Forward(self, input):

        # Begin Forward propagation
        input = torch.sigmoid(self.layerOne(torch.sigmoid(input)))
        input = torch.sigmoid(self.layerTwo(input))
        input = torch.sigmoid(self.layerThree(input))
        input = torch.sigmoid(self.layerFour(input))
        input = torch.sigmoid(self.layerFive(input))

        return input
    
    # Create function that will calculate the loss
    def Loss(self, predicted, target):

        return ((predicted - target) ** 2).mean()

    def Test(self, input):

        # Run the image through the network
        output = torch.sigmoid(self.layerOne(torch.sigmoid(input)))
        output = torch.sigmoid(self.layerTwo(output))
        output = torch.sigmoid(self.layerThree(output))
        output = torch.sigmoid(self.layerFour(output))
        output = torch.sigmoid(self.layerFive(output))

        return output
        
# Function to setup the dataset for letter and number recognition
def SetupData(digits, letters, ground_truths):

    # Create the ground truths for the data set by combining the letters and digits
    # and adding a one to the correct digit or number 
    final_truths = torch.zeros(digits.shape[0] + letters.shape[0], 36)
    for i in range(digits.shape[0] + letters.shape[0]):

        final_truths[i][int(ground_truths[i])] = 1

    # Shuffle together the letter and digit data concatenated with the repective 
    # ground truths and flatten the images
    dataset = torch.cat([digits, letters])
    flat_dataset = dataset.view((-1, 28 * 28)).float()

    # Torch's DataLoader used to shuffle the dataset to contain letters and digits arbitrarily
    permutation = torch.randperm(len(flat_dataset)).tolist()
    shuffled_images = torch.utils.data.Subset(flat_dataset, permutation)
    shuffled_letters = torch.utils.data.Subset(final_truths, permutation)
    imageDataLoader = torch.utils.data.DataLoader(shuffled_images, batch_size = 10, shuffle = False)
    groundTruthsDataLoader = torch.utils.data.DataLoader(shuffled_letters, batch_size = 10, shuffle = False)

    return imageDataLoader, groundTruthsDataLoader

# Create a function to write the specified data to a csv file
def WriteToCSV(fileName, data, rows, cols, bias):

    # Write the data to a csv file
    with open(fileName, mode = 'w') as output_file:
            output_file = csv.writer(output_file, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)

            for i in range(rows):

                for k in range(cols):

                    if bias:

                        output_file.writerow([str(float(data[k]))])
                    
                    else:

                        output_file.writerow([str(float(data[i][k]))])

# Function to write all the weights and biases in the neural network to a readable csv file
def ProcessData(neuralNet):

    # Call the WriteToCSV function for every weights and bias in the neural network 
    # each with a seperate output file name
    WriteToCSV("layerOneWeight", neuralNet.layerOne.weight, 512, 784, False)
    WriteToCSV("layerTwoWeight", neuralNet.layerTwo.weight, 512, 512, False)
    WriteToCSV("layerThreeWeight", neuralNet.layerThree.weight, 512, 512, False)
    WriteToCSV("layerFourWeight", neuralNet.layerFour.weight, 36, 512, False)
    WriteToCSV("layerOneBias", neuralNet.layerOne.bias, 1, 512, True)
    WriteToCSV("layerTwoBias", neuralNet.layerTwo.bias, 1, 512, True)
    WriteToCSV("layerThreeBias", neuralNet.layerThree.bias, 1, 512, True)
    WriteToCSV("layerFourBias", neuralNet.layerFour.bias, 1, 36, True)

if __name__ == "__main__":

    # read the mnist test and train dataset
    #mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Set the random seed for the data loader in order to ensure similar shuffling 
    # of the ground truths and the actual dataset
    torch.manual_seed(2809)
    torch.backends.cudnn.deterministic = True

    # Normalize the training elements and convert them to float
    # and convert the numpy array to torch tensor and define the learning rate
    x_train = x_train
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    learningRate = 0.3
    
    # read the emnist letter dataset to concatenate with the digits dataset
    letters, labels = extract_training_samples('letters')
    letters = torch.from_numpy(letters).float()
    labels = torch.from_numpy(labels).float() + 9 # Add 9 to differentiate from 0-9 digits

    ground_truths = torch.cat([y_train, labels])
    print(ground_truths.shape)

    # Create the dataloader that will contain all the data points shuffled through the SetupData function
    imageDataLoader, groundTruthsDataLoader = SetupData(x_train, letters, ground_truths)

    # Create a neural network instance and define the parameters list as well
    # for the backpropagation later on and setup the loss
    neuralNet = NeuralNetwork()
    params = list(neuralNet.parameters())
    criterion = nn.MSELoss()
    print(neuralNet)

    dataSet = next(iter(imageDataLoader))
    groundTruth = next(iter(groundTruthsDataLoader))

    # Loop through the number of epochs and feed Forward and 
    # also calculate and apply the gradients for the weights and biases
    for i in range(10):

        for k in range(24480):

            neuralNet.zero_grad()

            prediction = neuralNet.Forward(dataSet)
            loss = criterion(prediction, groundTruth)
            loss.backward()

            for layer in range(len(params)):

                # Updating the weights of the neural network
                params[layer].data.sub_(params[layer].grad.data * learningRate)

    # Prepare the test case data for torch network
    x_test = torch.from_numpy(x_test)
    x_test_flat = x_test.view((-1, 28 * 28)).float()

    # Run one test case through the newly trained neural network
    result = neuralNet.Test(x_test_flat[0])
    for i in range(36):

        print("Prediction of ", i,": ", result[i])

    print("Ground Truth: ", y_test[0])
    print("Loss: ", loss.item())

    # Run second test case on actual user handwriting
    handwrittenSix = plt.imread("ResizedImage.jpg")
    handwrittenSix = torch.from_numpy(handwrittenSix)

    handwrittenSixFlat = handwrittenSix.view((-1, 28 * 28)).float()
    result = neuralNet.Test(handwrittenSixFlat[0])
    for i in range(36):

        print("Prediction of ", i,": ", result[i])

    print("Ground Truth: 6")
    print("Loss: ", loss.item())

    #ProcessData(neuralNet)