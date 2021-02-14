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

class ConvNeuralNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.cnnLayers = nn.Sequential(

            # First convolutional layer
            nn.Conv2d(1, 4, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(4),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            # Second convolutional layer
            nn.Conv2d(4, 4, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(4),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))

        self.linearLayers = nn.Sequential(nn.Linear(4 * 7 * 7, 36))
    
    # feed-forward
    def forward(self, input):

        input = self.cnnLayers(input)
        input = input.view(input.size(0), -1)
        input = self.linearLayers(input)

        return input

        

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
    
    # define the data size, the batch size and the number of epochs
    dataSize = len(x_train) + len(letters)
    batchSize = 10
    epochs = 10

    # retrieve the corresponding dataloaders for the ground truths and the images datasets
    imageDataLoader, truthsDataLoader = PrepareData(tensor_letters, tensor_x_train, tensor_labels, tensor_y_train)

    # define the neural network
    neuralNet = ConvNeuralNet()
    neuralNet = neuralNet.float()

    # define the optimizer
    optimizer = torch.optim.Adam(neuralNet.parameters(), lr = 0.09)

    # !! begin the training !!
    imageBatch = next(iter(imageDataLoader))
    truthBatch = next(iter(truthsDataLoader))

    criterion = nn.MSELoss()

    print("shape of batch: ", imageBatch.shape)
    for i in range(epochs):

        for k in range(int(dataSize / batchSize)):

            # zero out the gradient
            optimizer.zero_grad()

            prediction = neuralNet.forward(imageBatch.float())
            loss = criterion(prediction, truthBatch)
            loss.backward()

            optimizer.step()

    # Prepare the test case data for torch network
    x_test = torch.from_numpy(x_test)

    # Run one test case through the newly trained neural network
    result = neuralNet.Test(x_test[0])
    for i in range(36):

        print("Prediction of ", i,": ", result[i])

    print("Ground Truth: ", y_test[0])
    print("Loss: ", loss.item())

    print(neuralNet)