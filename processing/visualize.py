import sys
import os
import utils
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def prev_slice(ax):
    volume = ax.volume
    ax.index = (ax.index-1) % volume.shape[0]
    ax.images[0].set_array(volume[ax.index])
   
def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index+1) % volume.shape[0]
    ax.images[0].set_array(volume[ax.index])
    

def process_key(event):  
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'a':
        prev_slice(ax)
    elif event.key == 'd':
        next_slice(ax)
    fig.canvas.draw()

def visSlice(volume):
    volume = np.array(volume)
    plt.ion()
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = 0
    ax.imshow(volume[ax.index])
    fig.canvas.mpl_connect('key_press_event',process_key)
    plt.show()
    _ = input("Press [Enter] for next example.")
    plt.close()
    

def main():
    uid = "1.3.6.1.4.1.14519.5.2.1.6279.6001.209269973797560820442292189762"
    centroids = [[150,150,150],[125,125,125]]
    for centroid in centroids:
        patient_img = utils.load_patient_images(uid, extension = "_i.png")
        volume = utils.get_cube_from_img(patient_img, centroid[0], centroid[1], centroid[2], 32) 
        visSlice(volume)
        

if __name__=="__main__":
    main()

