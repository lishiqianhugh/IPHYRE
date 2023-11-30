""" Define controller """
import torch
import torch.nn as nn
import pdb

class Controller(nn.Module):
    """ Controller """
    def __init__(self, latents, recurrents, n_actions):
        super().__init__()
        self.fc = nn.Linear(latents + recurrents, n_actions)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, *inputs, mask):
        cat_in = torch.cat(inputs, dim=1)
        out = self.fc(cat_in)
        out[0][mask==False] = -1e5
        return self.softmax(out)
