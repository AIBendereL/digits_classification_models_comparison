# Handwriting digits classification models comparison with Pytorch

This is my practice implementation of Mnist digits classification models, training and testing with neural network, convolutional neural network and pre-trained VGG16 using Pytorch. Combining with tracking model performance through training (loss, accuracy, total train time) to gain intuition how the  models work.

In the end, I get the train results of these models and make a comparison table.


### Dataset

**Mnist Digits**. (provided by Pytorch)


## Model architecture

VGG16 model architecture is provided by Pytorch. I change the last fully connected layer output to 10 nodes to match with the digits classification problem.

NN, CNN model is built by me. And the architecture is **COMPLETELY RANDOM**.


## Hyperparameters

All models are trained using the same set of hyperparameters.

```
- batch size : 32
- number of epochs: 50
- learning rate: 10^-3
```


## Result table

|| NN | CNN | VGG16 |
| ---- | ---- | ---- | ---- |
| Number of parameters | 669,706 | 3,797,962 | 138,357,544 |
| Average Train Loss | 0.204284 | 0.046307 | 0.000568 |
| Train Accuracy | 94.28% | 98.54% | 99.98% |
| Average Test Loss | 0.202827 | 0.086381 | 0.037905 |
| Test Accuracy | 94.01% | 97.28% | 99.34% |
| Total Train Time (mins) | 8.36 | 10.55 | 660.75 |


## File tree and description

```
|― CNN/ 
    |― cnn.py                       # CNN model class architecture
|― Mnist_digits_dataset/
    |― README.md                    # auto-download Mnist Digits dataset
|― Models/
    |― README.md                    # download pre-trained VGG16 result and weights 
    |― cnn_mnist_digits_50.pth      # CNN result
    |― nn_mnist_digits_50.pth       # NN result
|— NN/
    |― neural_network.py            # NN model class architecture
|— README.md
|— cnn_mnist_test.py                # CNN test script
|— cnn_mnist_train.py               # CNN train script
|— digits_classifier_test.py        # VGG16 test script
|— digits_classifier_train.py       # VGG16 train script
|— nn_mnist_test.py                 # NN test script
|— nn_mnist_train.py                # NN train script
```


### Run scripts

For train and test with VGG16, please download pre-trained VGG16 weights and result through the link in Models/README.md before running the scripts.

#### Train

You can train the model by running the corresponding train script through terminal. Then wait for the train process to finish. The result will be stored in Models/.

#### Test

You can quickly retrieve the model performance on test set by running the corresponding test script through terminal. The models are stored in Models/.

## Environment

Windows 10

Python 3.9.1

Pytorch 1.12.1 + CUDA 11.6

GPU Nvidia GeForce RTX 2060

