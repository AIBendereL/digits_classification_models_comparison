import torch
from torch import nn



#Model class:

class NN_v0(nn.Module):
    
    def __init__(self):
        super(NN_v0, self).__init__()
        
        self.flatten = nn.Flatten()
        
        self.classifier = nn.Sequential(
            nn.Linear(28*28, 512, bias = True),
            nn.ReLU(),
            nn.Linear(512, 512, bias = True),
            nn.ReLU(),
            nn.Linear(512, 10, bias = True)
        )


    def forward(self, images):
        logits = self.flatten(images)
        
        logits = self.classifier(logits)
        return logits
###





