import torch
import sys

boxes = torch.randn(3,5)
targets = torch.zeros((len(boxes), 6))
targets[:, 1:] = boxes
print(boxes.max(1)[0])
print(boxes.max(1)[0].argsort())