from torchvision.models import efficientnet_b0
from torch import nn


# # On définit le modèle
def get_efficientNet(N):
    model = efficientnet_b0(weights="EfficientNet_B0_Weights.DEFAULT")
    model._fc = nn.Linear(1280, N)
    return model

def get_efficientNet_Non_Trained(N):
    model = efficientnet_b0(weights=None)
    model._fc = nn.Linear(1280, N)
    return model