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

###



#Device:
device = "cuda:0" if torch.cuda.is_available() else "cpu"

###



#Modules:
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
    
    print(f"Average Test Loss: {test_loss}.")
    print(f"Accuracy: {accuracy}%.")


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
    MODEL_WEIGHTS_MNIST_DIGITS_PATH = "Models/vgg16_weights_mnist_digits.pth"
    
    model = vgg16()
    model.classifier[6] = nn.Linear(4096, 10, bias=True)
    
    model.load_state_dict(torch.load(MODEL_WEIGHTS_MNIST_DIGITS_PATH))
    
    model.to(device)
    
    
    print(model)
    print(f"Model Weights File: {MODEL_WEIGHTS_MNIST_DIGITS_PATH}.")
    ###
    


    #Test:
    loss_func = nn.CrossEntropyLoss()
    
    start = time.time()
    
    test_loop(test_loader, model, loss_func, device)
    
    end = time.time()
    total_time = (end - start) / 60
    
    print(f"Total time: {total_time:>5.2f} mins.\nDone.\n")
    
    ###
    
    