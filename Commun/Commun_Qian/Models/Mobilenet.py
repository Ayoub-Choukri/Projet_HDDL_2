from torchvision.models import mobilenet_v2
from torch import nn


# # On définit le modèle
def get_MobileNet(N, pretrained=True):
    model = mobilenet_v2(pretrained=pretrained)
    model.classifier[1] = nn.Linear(1280, N)
    return model
