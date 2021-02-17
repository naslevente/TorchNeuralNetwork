from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import matplotlib.pyplot as plt
import tensorflow as tf
from emnist import extract_training_samples

# Load letters and numbers data
letters, labels = extract_training_samples('letters')
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Change the data into torch tensors
tensor_x_train = torch.from_numpy(x_train)
tensor_letters = torch.from_numpy(letters)

count = 0
for i in range(len(letters)):

    if labels[i] == 1:

        count += 1

print("Numbers count: ", tensor_x_train.shape)
print("Letters count: ", tensor_letters.shape)
print("Number of datapoints per letter: ", count)