import torch
from torch import nn

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

from NN import neural_network

import time

import torchvision.transforms.functional as tf
from PIL import Image, ImageOps
import matplotlib.pyplot as plt


#Hyperparameter:

batch_size = 64
num_epochs = 10
learning_rate = 10 ** -3

###



#Device:

device = "cuda:0" if torch.cuda.is_available() else "cpu"

###



#Module:

def test_loop(test_loader, model, loss_func, device):
    size = len(test_loader.dataset)
    num_batches = len(test_loader)
    
    average_loss = 0
    accuracy = 0
    for batch, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            pred = model(images)
            loss = loss_func(pred, labels)
            
            average_loss += loss.item()
            accuracy += (pred.argmax(1) == labels).type(torch.float).sum().item()

        print(f"Current batch: {batch + 1}/{num_batches}.", end = "\r")
        
    average_loss /= num_batches
    accuracy /= size
    accuracy *= 100
    
    print(f"Average Test Loss: {average_loss:>7f}.")
    print(f"Test Accuracy: {accuracy:>3.2f}%.")

    

###



if __name__ == "__main__":
    
    #Transform:
    
    transform = transforms.Compose(
        [
            transforms.ToTensor()
        ]
    )
    
    ###
    
    
    
    #Data:
    
    train_set = datasets.MNIST(
        root = "Mnist_digits_dataset",
        train = True,
        download = True,
        transform = transform
    )
    
    test_set = datasets.MNIST(
        root = "Mnist_digits_dataset",
        train = False,
        download = True,
        transform = transform
    )
    
    
    train_loader = DataLoader(
        train_set,
        batch_size = batch_size,
        shuffle = True
    )
    
    test_loader = DataLoader(
        test_set,
        batch_size = batch_size,
        shuffle = True
    )
    
    ###
    
    
    
    #Model:
    MODEL_WEIGHTS_PATH = "Models/nn_mnist_digits_50.pth"

    model = neural_network.NN_v0()
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH))
    
    model.to(device)
    
    print(model)
    print(f"Model Weights File: {MODEL_WEIGHTS_PATH}.")
    
    
    ###
    
    
    
    #Test:
    loss_func = nn.CrossEntropyLoss()
    
    start = time.time()
    
    test_loop(test_loader, model, loss_func, device)
    
    end = time.time()
    total_time = end - start
    total_time /= 60
    
    print(f"Total time: {total_time:>3.2f}mins.")
    print("Done.")
    
    ###
    
             
    
    
    