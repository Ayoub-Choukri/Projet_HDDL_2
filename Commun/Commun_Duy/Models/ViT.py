import torch 
import torch.nn as nn 
import torch.nn.functional as F

from torchvision.models import vit_b_16, ViT_B_16_Weights

def load_vit_b_16(num_classes, pretrain = True, **kwargs): 
    if pretrain == True: 
        pretrained_vit_weights = ViT_B_16_Weights.DEFAULT
        vit = vit_b_16(weights=pretrained_vit_weights, **kwargs)    
    else:
        vit = vit_b_16(**kwargs)

    # transfer learning 
    # for param in vit.parameters():
    #     param.requires_grad = False

    vit.heads = nn.Linear(in_features=768, out_features=num_classes)
    # for param in vit.heads.parameters():
    #     param.requires_grad = True
    print("Nb of trainable params : ", sum(p.numel() for p in vit.parameters() if p.requires_grad))
    return vit 
