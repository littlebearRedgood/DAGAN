import torch
import numpy as np
import itertools


def quantRGB(bins,vmax=255,vmin=0):
    a = torch.linspace(vmin+((vmax-vmin)/(bins*2)), vmax-((vmax-vmin)/(bins*2)), bins)
    mat=torch.cartesian_prod(a,a,a)/vmax
    return mat.view(1,bins**3,3,1,1)


def quantL(bins,vmax,vmin):
    a = torch.linspace(vmin+((vmax-vmin)/(bins*2)), vmax-((vmax-vmin)/(bins*2)), bins)
    mat = a
    return mat.view(1,bins,1,1)


def quantAB(bins, vmax,vmin):
    a = torch.linspace(vmin+((vmax-vmin)/(bins*2)), vmax-((vmax-vmin)/(bins*2)), bins)
    mat=torch.cartesian_prod(a,a)
    return mat.view(1,bins**2,2,1,1)







