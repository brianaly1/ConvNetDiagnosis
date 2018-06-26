import sys
import os
import utils
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def visVolume(volume):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(volume)
    plt.show()
    plt.close()

def visSlice(volume):
    for image in volume:
        plt.imshow(image, cmap=plt.cm.gray)
        plt.show()
        plt.close()

def main():
    volume = utils.load_cube_img("/home/alyb/Data/TrainData/LIDC/1.3.6.1.4.1.14519.5.2.1.6279.6001.100621383016233746780170740405_4_4_1_pos.png",8,8,64)
    visSlice(volume)

if __name__=="__main__":
    main()

