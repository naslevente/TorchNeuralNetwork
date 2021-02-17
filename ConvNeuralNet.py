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
from emnist import extract_test_samples
import csv
import torch.nn.functional as Func

class ConvNeuralNet(nn.Module):

    def __init__(self):

        super(ConvNeuralNet, self).__init__()

        # input image with 1 channel with 6 output channels and a 3x3 kernel filter
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)

        self.linear1 = nn.Linear(16 * 5 * 5, 120)
        self.linear2 = nn.Linear(120, 84)

        # change 10 to 36 if the network is training on both the letters and the numbers
        self.linear3 = nn.Linear(84, 10)
    
    def forward(self, input):

        input = Func.max_pool2d(torch.sigmoid(self.conv1(input)), (2, 2))
        input = Func.max_pool2d(torch.sigmoid(self.conv2(input)), 2)

        input = input.view(-1, self.num_flat_features(input))
        input = torch.sigmoid(self.linear1(input))
        input = torch.sigmoid(self.linear2(input))
        input = self.linear3(input)

        return input
        
    def num_flat_features(self, input):

        size = input.size()[1:]
        num_features = 1
        for i in size:

            num_features *= i
        
        return num_features


# function to prepare and randomize the data
def PrepareData(letters, numbers, lettersTruths, numbersTruths):

    # combine the letters and numbers ground truths and images datasets
    complete_dataset = torch.cat([letters, numbers])
    complete_truths = torch.cat([lettersTruths, numbersTruths])

    size = letters.shape[0] + numbers.shape[0]
    final_truths = torch.zeros(size, 36)
    for i in range(size):

        final_truths[i][int(complete_truths[i])] = 1

    # shuffle the images and truths datasets in the same order
    permutation = torch.randperm(len(complete_dataset)).tolist()
    shuffled_images = torch.utils.data.Subset(complete_dataset, permutation)
    shuffled_truths = torch.utils.data.Subset(final_truths, permutation)

    # load the data into dataloaders for simplicity
    imageDataLoader = torch.utils.data.DataLoader(shuffled_images, batch_size = 10, shuffle = False)
    groundTruthsDataLoader = torch.utils.data.DataLoader(shuffled_truths, batch_size = 10, shuffle = False)

    return imageDataLoader, groundTruthsDataLoader

if __name__ == "__main__":

    # Load letters and numbers data
    letters, labels = extract_training_samples('letters')
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Change the data and truths into torch tensors and apply the unsqueeze method
    # to the datasets to add another dimension for the number of channels
    tensor_x_train = torch.from_numpy(x_train).unsqueeze(1)
    tensor_letters = torch.from_numpy(letters).unsqueeze(1)
    tensor_y_train = torch.from_numpy(y_train)
    tensor_labels = torch.from_numpy(labels) + 9

    if torch.cuda.is_available():

        tensor_x_train = tensor_x_train.cuda()
        tensor_letters = tensor_letters.cuda()
        tensor_y_train = tensor_y_train.cuda()
        tensor_labels = tensor_labels.cuda()
    
    # define the data size, the batch size, learning rate and the number of epochs
    dataSize = len(x_train)
    batchSize = 10
    learningRate = 0.2
    epochs = 15

    # convert the truth data into applicable tensor data
    size = tensor_x_train.shape[0]
    final_truths = torch.zeros(size, 10)
    for i in range(size):

        final_truths[i][int(y_train[i])] = 1

    # retrieve the corresponding dataloaders for the ground truths and the images datasets
    #imageDataLoader, truthsDataLoader = PrepareData(tensor_letters, tensor_x_train, tensor_labels, tensor_y_train)
    permutation = torch.randperm(len(tensor_x_train)).tolist()
    shuffled_images = torch.utils.data.Subset(tensor_x_train, permutation)
    shuffled_truths = torch.utils.data.Subset(final_truths, permutation)

    imageDataLoader = torch.utils.data.DataLoader(shuffled_images, batch_size = 10, shuffle = False)
    truthsDataLoader = torch.utils.data.DataLoader(shuffled_truths, batch_size = 10, shuffle = False)

    # define the neural network
    neuralNet = ConvNeuralNet()
    neuralNet = neuralNet.float()

    # define the optimizer
    #optimizer = torch.optim.Adam(neuralNet.parameters(), lr = 0.09)

    # !! begin the training !!

    # create iterators for the dataloaders
    imageBatchIterator = iter(imageDataLoader)
    truthBatchIterator = iter(truthsDataLoader)

    # define the loss function used
    criterion = nn.MSELoss()

    # setup the test cases 
    x_test = torch.from_numpy(x_test)

    testCase = x_test[0]
    testCase = testCase.unsqueeze(0).unsqueeze(0).float()

    # all the weights of the neural network
    params = list(neuralNet.parameters())
    print("initial weights: ", params[0])

    for i in range(epochs):

        # pick out the first image and respective truth
        imageBatch = next(imageBatchIterator)
        truthBatch = next(truthBatchIterator)

        for k in range(int(dataSize / batchSize)):

            # zero out the gradient
            #optimizer.zero_grad()
            neuralNet.zero_grad()

            prediction = neuralNet.forward(imageBatch.float())
            loss = criterion(prediction, truthBatch)
            loss.backward()

            #optimizer.step()
            # update the weights of the network
            for i in neuralNet.parameters():

                i.data.sub_(i.grad.data * learningRate)

            try:

                # set the current batches of data using iterator
                imageBatch = next(imageBatchIterator)
                truthBatch = next(truthBatchIterator)
            
            except StopIteration:

                imageBatchIterator = iter(imageDataLoader)
                truthBatchIterator = iter(truthsDataLoader)


    # Run one test case through the newly trained neural network
    result = neuralNet.forward(testCase)

    print("result: ", result)
    print("loss: ", loss.item())

    '''
    for i in range(36):

        print("Prediction of ", i,": ", result[i])

    print("Ground Truth: ", y_test[0])
    #print("Loss: ", loss.item())
    '''

    print(neuralNet)