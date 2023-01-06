import torch
from torch import nn

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.data import DataLoader

from NN import neural_network

import time

#Hyperparameters:

batch_size = 32
num_epochs = 50
learning_rate = 10 ** -3

###



#Device:

device = "cuda:0" if torch.cuda.is_available() else "cpu"

###



#Module:

def train_loop(train_loader, model, loss_func, optimizer, device):
    size = len(train_loader.dataset)
    num_batches = len(train_loader)
    
    train_loss = 0
    accuracy = 0
    for batch, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        pred = model(images)
        loss = loss_func(pred, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        accuracy += (pred.argmax(1) == labels).type(torch.float).sum().item()
        
        if batch % 100 == 0:
            current_loss = loss.item()
            current_example = batch * len(images)
            
            print(f"Train loss: {current_loss:>7f} \t [{current_example:>5d}/{size:>5d}].")


    train_loss /= num_batches
    accuracy /= size
    accuracy *= 100
    
    
    print(f"Average Train Loss: {train_loss:>7f}.")
    print(f"Train Accuracy: {accuracy:3.2f}%.")
    

def test_loop(test_loader, model, loss_dunc, device):
    size = len(test_loader.dataset)
    num_batches = len(test_loader)
    
    test_loss = 0
    accuracy = 0
    
    with torch.no_grad():
        for batch, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            pred = model(images)
            loss = loss_func(pred, labels)
            
            test_loss += loss.item()
            accuracy += (pred.argmax(1) == labels).type(torch.float).sum().item()
            
            print(f"Current batch: {batch + 1}/{num_batches}.", end = "\r")
        
    
    test_loss /= num_batches
    accuracy /= size
    accuracy *= 100
    
    
    print(f"Average Test Loss: {test_loss:>7f}.")
    print(f"Test Accuracy: {accuracy:3.2f}%.")

    
###



if __name__ == "__main__":
    
    #Transforms:
    
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    
    ###
    
    
    
    #Data:
    
    MNIST_DIGITS_PATH = "Mnist_digits_dataset"
    
    train_set = datasets.MNIST(
        root = MNIST_DIGITS_PATH,
        train = True,
        download = True,
        transform = transform
    )
    
    test_set = datasets.MNIST(
        root = MNIST_DIGITS_PATH,
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
    
    model = neural_network.NN_v0()
    model.to(device)
    
    print(model)
    
    ###
    
    
    
    #Train:
    NN_MNIST_DIGITS_PATH = "Models/nn_mnist_digits_40.pth"
    
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
    
    start = time.time()
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1} / {num_epochs}: \n--------------------")
        
        train_loop(train_loader, model, loss_func, optimizer, device)
        test_loop(test_loader, model, loss_func, device)
        
        torch.save(model.state_dict(), NN_MNIST_DIGITS_PATH)

    end = time.time()
    total_time = (end - start) / 60
    
    print(f"Total time: {total_time:>5.2f} mins. \nDone.\n")
    ###
    
    
    
    
    
    
    