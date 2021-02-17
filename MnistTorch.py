from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import matplotlib.pyplot as plt
import tensorflow as tf
from emnist import extract_training_samples
from emnist import extract_test_samples

'''
mnist = MNIST("data", download=True)


split = int(0.8 * len(mnist))
index_list = list(range(len(mnist)))
train_idx, valid_idx = index_list[:split], index_list[split:]

split = int(0.8 * len(mnist))
index_list = list(range(len(mnist)))
train_idx, valid_idx = index_list[:split], index_list[split:]

tr_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(valid_idx)

trainloader = DataLoader(mnist, batch_size=256, sampler=tr_sampler)
validloader = DataLoader(mnist, batch_size=256, sampler=val_sampler)

plt.imshow(trainloader[0])


threes = mnist.data[(mnist.targets == 3)]
sevens = mnist.data[(mnist.targets == 7)]

target = torch.tensor([1]*len(threes)+[2]*len(sevens))
target.shape

plt.imshow(threes[0])
plt.show()

combined_data = torch.cat([threes, sevens])
print("Torch data shape: ", target.shape)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print(y_train[0])

print(type(x_train))
x_train = torch.from_numpy(x_train)
print("Tensorflow data shape: ", x_train.shape)

flat_imgs = x_train.view((-1, 28 * 28))
print(flat_imgs.shape)

ground_truths = torch.zeros(60000, 10)
for i in range(len(y_train)):

    ground_truths[i][y_train[i]] = 1

print(ground_truths.shape)


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
#plt.imshow(x_train[0])
print(x_train[0][13])

images, labels = extract_training_samples('letters')
print(labels[0])
#plt.imshow(images[0])
print(len(images))
print(images[0][13])
#plt.show()

x_train = x_train
x_train = torch.from_numpy(x_train)
flat_imgs = x_train.view((-1, 28 * 28)).float()
print("Shape of the flattened images: ", flat_imgs.shape)

print("Training data shape: ", x_train[0].shape)
print(y_train[0])

images = torch.from_numpy(images)
complete_dataset = torch.cat([x_train, images])
flat_dataset = complete_dataset.view((-1, 28 * 28)).float()

loader = torch.utils.data.DataLoader(complete_dataset, shuffle = True)
loader2 = torch.utils.data.DataLoader(flat_dataset, shuffle = True)

dataset = next(iter(loader2))
print(dataset.shape)


for i in range(20):

    dataset = next(iter(loader))
    print(dataset[0].shape)

    numpyImage = dataset[0].numpy()
    plt.imshow(numpyImage)
    plt.show()

Some scrappedd code from the torch neural network
flat_imgs = x_train.view((-1, 28 * 28)).float()
    learningRate = 0.3

    ground_truths = torch.zeros(60000, 10)
    for i in range(len(y_train)):

        ground_truths[i][y_train[i]] = 1

# Function to setup the dataset for letter and number recognition
def SetupData(digits, letters, ground_truths):

    # Create the ground truths for the data set by combining the letters and digits
    # and adding a one to the correct digit or number 
    final_truths = torch.zeros(digits.shape[0] + letters.shape[0], 36)
    for i in range(digits.shape[0] + letters.shape[0]):

        final_truths[i][ground_truths[i]] = 1

    # Shuffle together the letter and digit data concatenated with the repective 
    # ground truths and flatten the images
    dataset = torch.cat([digits, letters])
    flat_dataset = complete_dataset.view((-1, 28 * 28)).float()
    complete_dataset = torch.utils.data.ConcatDataset(flat_dataset, final_truths)

    # Torch's DataLoader used to shuffle the dataset to contain letters and digits arbitrarily
    dataLoader = torch.utils.data.DataLoader(flat_dataset, batch_size = 10, shuffle = True)

    return dataLoader
'''

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
    #flat_dataset = dataset.view((-1, 28 * 28)).float()

    # Torch's DataLoader used to shuffle the dataset to contain letters and digits arbitrarily
    permutation = torch.randperm(len(dataset)).tolist()
    shuffled_images = torch.utils.data.Subset(dataset, permutation)
    shuffled_letters = torch.utils.data.Subset(final_truths, permutation)
    imageDataLoader = torch.utils.data.DataLoader(shuffled_images, batch_size = 10, shuffle = False)
    groundTruthsDataLoader = torch.utils.data.DataLoader(shuffled_letters, batch_size = 10, shuffle = False)

    return imageDataLoader, groundTruthsDataLoader

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
print("Mnist shape: ", x_train.shape)
y_train = torch.from_numpy(y_train)
    
# read the emnist letter dataset to concatenate with the digits dataset
letters, labels = extract_training_samples('letters')
letters = torch.from_numpy(letters)
print("Emnist shape: ", letters.shape)
complete_dataset = torch.cat([x_train, letters])
flat_dataset = complete_dataset.view((-1, 28 * 28)).float()
print("Complete dataset shape: ", complete_dataset.shape)
#letters_flat = letters.view((-1, 28 * 28)).float()
letters_flat = torch.flatten(letters)
print("Flattened shape: ", letters_flat.shape)
labels = torch.from_numpy(labels).float() + 9 # Add 9 to differentirate from 0-9 digits
print(labels[0])

ground_truths = torch.cat([y_train, labels])
print(ground_truths.shape)

# Create the dataloader that will contain all the data points shuffled through the SetupData function
imageDataLoader, groundTruthsDataLoader = SetupData(x_train, letters, ground_truths)

# Create a tensor with all the shuffled data points from the data loader
dataset = next(iter(imageDataLoader))
groundTruth = next(iter(groundTruthsDataLoader))

'''
for i in range(10):

    #print(dataset)

    #print(groundTruth.shape)
    #print(type(groundTruth))

    numpyImage = dataset[i].numpy()
    print(groundTruth[i])
    plt.imshow(numpyImage)
    plt.show()

    dataset = next(iter(imageDataLoader))
    groundTruth = next(iter(groundTruthsDataLoader))
'''

plt.imshow(x_test[0])
#print("First Image: ", x_test[0])
print(x_test[0].shape)
plt.show()

x_test = torch.from_numpy(x_test)
x_test_flat = x_test.view((-1, 28 * 28)).float()

#print(x_test_flat[0])

handwrittenSix = plt.imread("ResizedImage.jpg")
handwrittenSix = torch.from_numpy(handwrittenSix)

handwrittenSixFlat = handwrittenSix.view(-1, 28 * 28).float()

#print(handwrittenSixFlat[0])

test_letters, test_labels = extract_test_samples('letters')
print("Number of letters dimension: ", test_letters.shape)
test_letters = torch.from_numpy(test_letters).float()
test_labels = torch.from_numpy(test_labels).float()
print(test_labels[3])
plt.imshow(test_letters[3])
plt.show()
complete_dataset = torch.cat([test_letters, test_letters])

test_letters_flat = complete_dataset.view((-1, 28 * 28)).float()
print(test_letters_flat.shape)
print(test_letters_flat[4])

'''
for i in range(10):

    print(y_test[i])
'''
#print("Shape of final data set: ", dataset.shape)
#print(dataset.shape[0])

    