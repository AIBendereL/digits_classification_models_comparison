import torch
from torch import nn

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import vgg16, VGG16_Weights

from torch.utils.data import DataLoader

import time



#Hyper Parameters:
batch_size = 32
num_epochs = 50
learning_rate = 10 ** -3

###



#Device:
device = "cuda:0" if torch.cuda.is_available() else "cpu"

###



#Modules:
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
            
            print(f"Train Loss: {current_loss:>7f} \t [{current_example:>5d}/{size:>5d}].")


    train_loss /= num_batches
    accuracy /= size
    accuracy *= 100
    
    
    print(f"Average Train Loss: {train_loss:>7f}.")
    print(f"Train Accuracy: {accuracy:3.2f}%.")


def test_loop(test_loader, model, loss_func, device):
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
            
            test_loss += loss
            accuracy += (pred.argmax(1) == labels).type(torch.float).sum().item()
            
            print(f"Current Batch: {batch + 1}/{num_batches}.", end = "\r")
            
        
    test_loss /= num_batches
    accuracy /= size
    accuracy *= 100
    
    print(f"Average Test Loss: {test_loss:>7f}.")
    print(f"Test Accuracy: {accuracy:3.2f}%.")
    
        
###



if __name__ == "__main__":
    
    #Data:
    MNIST_DIGITS_PATH = "Mnist_digits_dataset"
    weights = VGG16_Weights.IMAGENET1K_V1
    preprocess_images = weights.transforms()
    
    
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x:
                    x.repeat(3, 1, 1)
            ),
            preprocess_images
        ]
    )
    
    
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
    MODEL_WEIGHTS_PATH = "Models/vgg16_weights.pth"
    
    model = vgg16()
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH))
    
    model.classifier[6] = nn.Linear(4096, 10, bias=True)
    
    model.to(device)
    
    
    print(model)
    
    ###
    
    
    
    #Train:
    MODEL_WEIGHTS_MNIST_DIGITS_PATH = "Models/vgg16_mnist_digits_50.pth"
    
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
    
    
    start = time.time()
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1} / {num_epochs}:\n--------------------")
        
        train_loop(train_loader, model, loss_func, optimizer, device)
        test_loop(test_loader, model, loss_func, device)
        
        torch.save(model.state_dict(), MODEL_WEIGHTS_MNIST_DIGITS_PATH)

        
    end = time.time()
    total_time = (end - start) / 60
    
    print(f"Total time: {total_time:>5.2f} mins.\nDone.\n")
    
    ###
    