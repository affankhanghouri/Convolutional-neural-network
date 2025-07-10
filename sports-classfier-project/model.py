import torch
import torch.nn as nn
from torchvision import models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pre-trained VGG16
model = models.vgg16(pretrained=True)

# Modify classifier to match 100 output classes
model.classifier = nn.Sequential(
    nn.Linear(25088, 4096),
    nn.BatchNorm1d(4096),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),

    nn.Linear(4096, 3000),
    nn.BatchNorm1d(3000),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),

    nn.Linear(3000, 614),
    nn.BatchNorm1d(614),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),

    nn.Linear(614, 100)
)

# Load trained weights
model.load_state_dict(torch.load("sports_claassifier.pth", map_location=device))
model.to(device)
model.eval()
