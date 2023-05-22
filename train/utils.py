# System

# Libs
import torch
# Our sources


def get_bounding_box(idxcache):
    minX = torch.min(idxcache[0]).item()
    maxX = torch.max(idxcache[0]).item()
    minY = torch.min(idxcache[1]).item()
    maxY = torch.max(idxcache[1]).item()
    return (minX,minY,maxX,maxY)