import torch
import torch.nn as nn
from torchvision import models

def build_model(num_classes=8, device='cpu'):
    # Load pretrained ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Freeze all layers — we don't want to overwrite ImageNet knowledge
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the final layer with one for our 8 classes
    # This is the only layer that will actually train
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model.to(device)