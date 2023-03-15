import torch
from torch import nn

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

from CNN import cnn

import time



#Hyperparameter:

batch_size = 64
num_epochs = 20
learning_rate = 10 ** -3


###



#Device:

device = "cuda:0" if torch.cuda.is_available() else "cpu"

###



#Module:
def test_loop(test_loader, model, loss_func, device):
    size = len(test_loader.dataset)
    num_batches = len(test_loader)

    test_loss = 0
    accuracy = 0
    
    for batch, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            pred = model(images)
            loss = loss_func(pred, labels)
            
            test_loss += loss.item()
            accuracy += (pred.argmax(1) == labels).type(torch.float).sum().item()
            
        print(f"Current batch: {batch + 1}/{num_batches}.", end = "\r")
        
    test_loss /= num_batches
    accuracy /= size
    accuracy *= 100
    
    print(f"Average Test Loss: {test_loss:>7f}.")
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
    CNN_MODEL_WEIGHTS_PATH = "Models/cnn_mnist_digits_50.pth"
    
    model = cnn.CNN_v0()
    model.load_state_dict(torch.load(CNN_MODEL_WEIGHTS_PATH))
    
    model.to(device)
    
    print(model)
    print(f"Model Weights File: {CNN_MODEL_WEIGHTS_PATH}.")


    ###



    #Test:
    loss_func = nn.CrossEntropyLoss()

    start = time.time()
    
    test_loop(test_loader, model, loss_func, device)
    
    end = time.time()
    total_time = end - start
    total_time /= 60
    
    print(f"Total Time: {total_time:>3.2f} mins.\nDone\n")
    
    ###




