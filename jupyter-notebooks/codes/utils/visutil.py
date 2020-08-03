#--- Visualize result using matplotlib ---#
import torch
import torchvision as tcvs
import matplotlib.pyplot as plt

from utils.osutils import *
from utils.imutils import *
from utils.misc import *

def make_grid(inputs):
    return tcvs.utils.make_grid(inputs)

def show_img(img):
    plt.imshow(im_to_numpy(img))
    plt.show()

def show_heatmaps_all(hms):
    assert hms.dim() == 3

    hm_all = torch.zeros(hms.size(1), hms.size(2))
    for p in range(hms.size(0)):
        hm_all += hms[p]
    plt.imshow(hm_all.clamp(max=1, min=0), cmap='gray')
    #plt.colorbar()
    plt.show()

def show_heatmap(hm):
    assert hm.dim() == 2

    plt.imshow(hm, cmap='gray')
    #plt.colorbar()
    plt.show()

def draw_pose(img, joints):
    pass