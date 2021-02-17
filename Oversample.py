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


letters, labels = extract_training_samples('letters')
torch_letters = torch.from_numpy(letters).float()
torch_labels = torch.from_numpy(labels) + 9
print(torch_labels.shape)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print(len(x_train))

new_size = 86400
new_labels = torch.zeros(new_size)
new_dataset = torch.zeros(new_size, 28, 28)
print(new_dataset.shape)

count = 0
accepted_letters = [10, 11, 12, 13, 20, 22, 25, 28, 30, 33, 34, 35]

isFull = False
i = 0
acount = 0
while i < len(labels) and not(isFull):

    if torch_labels[i] == 10:

        acount += 1

    if torch_labels[i] in accepted_letters:

        for k in range(3):

            new_dataset[count + k] = torch_letters[i]
            new_labels[count + k] = accepted_letters.index(torch_labels[i]) + 10
            
        count += 3

    if count >= new_size:

        isFull = True
            
    i += 1

print("total data points: ", count)
print("data points per letter: ", acount)

print(new_dataset.shape)
for i in range(10):

    letter = new_dataset[i].numpy()
    print("Ground truth: ", new_labels[i])
    plt.imshow(letter)
    plt.show()

'''
for i in range(5):

    letter = new_dataset[i]
    plt.imshow(letter)
    plt.show()
'''