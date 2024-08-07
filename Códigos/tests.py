from torchvision.models import convnext_tiny
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import torch.nn as nn
import torch

model = convnext_tiny(weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
submodel = list(model.children())[:-2]
print(len(list(submodel[-1])))
sequential = list(submodel[-1].children())
excluded_sequential = sequential.pop()
submodel[-1] = torch.nn.Sequential(*sequential)
