import torch
from torch import nn



#Model class

class CNN_v0(nn.Module):
    
    def __init__(self):
        super(CNN_v0, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0)),
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),  
            nn.ReLU()
            # nn.Conv2d(128, 128, (3, 3), (1, 1), (1, 1))
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        self.flatten = nn.Flatten()
        
        self.classifier = nn.Sequential(
            nn.Linear(3136, 1024, bias=True),
            nn.ReLU(),
            # nn.Linear(2048, 1024, bias=True),
            # nn.ReLU(),
            nn.Linear(1024, 512, bias=True),
            nn.ReLU(),
            nn.Linear(512, 10, bias=True)
        )
        
    
    def forward(self, images):
        logits = self.features(images)
        logits = self.avgpool(logits)
        logits = self.flatten(logits)
        logits = self.classifier(logits)
        return logits

###




